import os
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///annotation.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ==================== 数据库模型 ====================

class TextFile(db.Model):
    """文本文件表"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.Integer, default=0)  # 0:未开始 1:进行中 2:已完成
    
    # 关联标注记录
    annotations = db.relationship('Annotation', backref='file', lazy=True, cascade='all, delete-orphan')


class Annotation(db.Model):
    """标注记录表"""
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('text_file.id'), nullable=False)
    tag_type = db.Column(db.String(50), nullable=False)  # 命名实体/情感标注
    start_index = db.Column(db.Integer, nullable=False)
    end_index = db.Column(db.Integer, nullable=False)
    selected_text = db.Column(db.String(200), nullable=False)
    label = db.Column(db.String(50), nullable=False)  # 人名/地名/组织/时间/情感
    created_time = db.Column(db.DateTime, default=datetime.utcnow)


class KnowledgeEntity(db.Model):
    """知识库实体表（替代JSON文件）"""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), unique=True, nullable=False)
    label = db.Column(db.String(50), nullable=False)
    source = db.Column(db.String(50), default='manual')  # manual/auto
    created_time = db.Column(db.DateTime, default=datetime.utcnow)


# 创建数据库表
with app.app_context():
    db.create_all()


# ==================== 知识库操作函数 ====================

def add_to_knowledge_base(text, label, source='manual'):
    """将实体添加到知识库"""
    existing = KnowledgeEntity.query.filter_by(text=text).first()
    if not existing:
        entity = KnowledgeEntity(text=text, label=label, source=source)
        db.session.add(entity)
        db.session.commit()
        return True
    return False


def get_knowledge_entities():
    """获取所有知识库实体"""
    return KnowledgeEntity.query.order_by(KnowledgeEntity.text).all()


def get_knowledge_dict():
    """获取知识库字典格式（用于智能标注）"""
    entities = KnowledgeEntity.query.all()
    return {e.text: e.label for e in entities}


# ==================== 辅助函数 ====================

def get_pos_cn(flag):
    """将jieba词性标记映射为中文名称和颜色类别"""
    mapping = {
        'n': ('名词', 'primary'),
        'nr': ('人名', 'info'),
        'ns': ('地名', 'success'),
        'nt': ('机构', 'danger'),
        'nz': ('其他专名', 'primary'),
        'v': ('动词', 'success'),
        'vd': ('副动词', 'success'),
        'vn': ('名动词', 'success'),
        'a': ('形容词', 'warning'),
        'ad': ('副形词', 'warning'),
        'd': ('副词', 'secondary'),
        'm': ('数词', 'dark'),
        'q': ('量词', 'dark'),
        'r': ('代词', 'secondary'),
        'p': ('介词', 'secondary'),
        'c': ('连词', 'secondary'),
        'u': ('助词', 'secondary'),
        'xc': ('虚词', 'secondary'),
        'w': ('标点', 'light'),
        'x': ('标点', 'light'),
        't': ('时间', 'info')
    }
    return mapping.get(flag, ('其他', 'light'))


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """首页 - 任务管理"""
    files = TextFile.query.order_by(TextFile.upload_time.desc()).all()
    return render_template('index.html', files=files)


@app.route('/upload', methods=['POST'])
def upload_file():
    """上传文件"""
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '' or not file:
        return redirect(url_for('index'))
    
    # 直接读取文件内容，不保存文件到磁盘
    try:
        content = file.read().decode('utf-8')
    except UnicodeDecodeError:
        file.seek(0)  # 重置文件指针
        content = file.read().decode('gbk')
    
    new_file = TextFile(filename=file.filename, content=content)
    db.session.add(new_file)
    db.session.commit()
    
    return redirect(url_for('index'))


@app.route('/manual_input', methods=['POST'])
def manual_input():
    """手动输入文本"""
    task_name = request.form.get('task_name', '').strip()
    text_content = request.form.get('text_content', '').strip()
    
    if not task_name or not text_content:
        return redirect(url_for('index'))
    
    # 确保文件名以.txt结尾
    if not task_name.endswith('.txt'):
        task_name += '.txt'
    
    new_file = TextFile(filename=task_name, content=text_content)
    db.session.add(new_file)
    db.session.commit()
    
    return redirect(url_for('index'))


@app.route('/annotate/<int:file_id>')
def annotate(file_id):
    """标注页面"""
    file_data = TextFile.query.get_or_404(file_id)
    annotations = Annotation.query.filter_by(file_id=file_id).all()
    
    # 生成分词展示数据
    segments = []
    words = pseg.cut(file_data.content)
    for word, flag in words:
        cn_pos, color_style = get_pos_cn(flag)
        segments.append({
            'word': word,
            'pos': cn_pos,
            'style': color_style
        })
    
    return render_template('annotate.html', file=file_data, annotations=annotations, segments=segments)


@app.route('/auto_annotate/<int:file_id>')
def auto_annotate(file_id):
    """智能自动标注"""
    file_data = TextFile.query.get_or_404(file_id)
    content = file_data.content
    
    # 清除旧标注
    Annotation.query.filter_by(file_id=file_id).delete()
    
    # 记录已标注位置，避免重叠
    annotated_positions = set()
    
    # ========== 1. 优先使用知识库进行标注 ==========
    knowledge_dict = get_knowledge_dict()
    # 按长度降序排列，优先匹配较长的实体
    sorted_entities = sorted(knowledge_dict.items(), key=lambda x: len(x[0]), reverse=True)
    
    for entity_text, entity_label in sorted_entities:
        start = 0
        while True:
            start = content.find(entity_text, start)
            if start == -1:
                break
            
            end = start + len(entity_text)
            
            # 检查是否与已标注位置重叠
            is_overlapping = any(pos in annotated_positions for pos in range(start, end))
            
            if not is_overlapping:
                tag_type = "命名实体" if entity_label in ['人名', '地名', '组织', '时间'] else "情感标注"
                new_ann = Annotation(
                    file_id=file_id,
                    tag_type=tag_type,
                    start_index=start,
                    end_index=end,
                    selected_text=entity_text,
                    label=entity_label
                )
                db.session.add(new_ann)
                
                # 记录已标注位置
                annotated_positions.update(range(start, end))
            
            start = end
    
    # ========== 2. 使用jieba分词进行实体识别 ==========
    words = pseg.cut(content)
    current_index = 0
    
    for word, flag in words:
        # 跳过已在知识库中处理过的词
        if word in knowledge_dict:
            current_index = content.find(word, current_index)
            if current_index != -1:
                current_index += len(word)
            continue
        
        tag_type = None
        label = None
        
        # 根据词性判断实体类型
        if flag == 'nr':
            tag_type, label = "命名实体", "人名"
        elif flag == 'ns':
            tag_type, label = "命名实体", "地名"
        elif flag == 'nt':
            tag_type, label = "命名实体", "组织"
        elif flag == 't':
            tag_type, label = "命名实体", "时间"
        
        # 情感词识别（形容词、副词）
        if flag in ['a', 'ad', 'd'] and len(word) > 1:
            try:
                sentiment_score = SnowNLP(word).sentiments
                if sentiment_score > 0.6 or sentiment_score < 0.4:
                    tag_type, label = "情感标注", "情感"
            except:
                pass
        
        # 查找位置并添加标注
        start = content.find(word, current_index)
        
        if tag_type and start != -1:
            end = start + len(word)
            is_overlapping = any(pos in annotated_positions for pos in range(start, end))
            
            if not is_overlapping:
                new_ann = Annotation(
                    file_id=file_id,
                    tag_type=tag_type,
                    start_index=start,
                    end_index=end,
                    selected_text=word,
                    label=label
                )
                db.session.add(new_ann)
                annotated_positions.update(range(start, end))
        
        # 更新索引
        if start != -1:
            current_index = start + len(word)
        else:
            current_index += len(word)
    
    # ========== 3. 使用正则表达式识别日期 ==========
    date_patterns = [
        r'(\d{4})(?:年|-|/)(\d{1,2})(?:月|-|/)(\d{1,2})(?:日)?',
        r'(\d{1,2})(?:月|-|/)(\d{1,2})(?:日)?',
        r'(\d{4})(?:年)(\d{1,2})(?:月)',
        r'(?:今年|去年|明年|今天|明天|昨天)',
        r'(?:本月|上月|下月)',
        r'(?:星期|周)(?:一|二|三|四|五|六|日|天)'
    ]
    
    combined_pattern = '|'.join(date_patterns)
    
    for match in re.finditer(combined_pattern, content):
        start = match.start()
        end = match.end()
        date_str = match.group()
        
        is_overlapping = any(pos in annotated_positions for pos in range(start, end))
        
        if not is_overlapping:
            new_ann = Annotation(
                file_id=file_id,
                tag_type="命名实体",
                start_index=start,
                end_index=end,
                selected_text=date_str,
                label="时间"
            )
            db.session.add(new_ann)
            annotated_positions.update(range(start, end))
    
    # 更新文件状态
    file_data.status = 1
    db.session.commit()
    
    return redirect(url_for('annotate', file_id=file_id))


@app.route('/stats')
def stats():
    """统计页面"""
    from sqlalchemy import func
    
    total_files = TextFile.query.count()
    total_anns = Annotation.query.count()
    
    # 获取各标注类型的统计
    stats_query = db.session.query(
        Annotation.label, 
        func.count(Annotation.id)
    ).group_by(Annotation.label).all()
    
    stats_dict = dict(stats_query)
    
    # 确保所有标注类型都有数据
    required_labels = ['人名', '地名', '组织', '时间', '情感']
    complete_stats = [(label, stats_dict.get(label, 0)) for label in required_labels]
    
    # 计算平均标注数
    avg_anns = round(total_anns / total_files) if total_files > 0 else 0
    
    # 准备图表数据
    labels = [s[0] for s in complete_stats]
    counts = [s[1] for s in complete_stats]
    
    return render_template('stats.html',
                           stats_data=complete_stats,
                           labels=labels,
                           counts=counts,
                           total_files=total_files,
                           total_anns=total_anns,
                           avg_anns=avg_anns)


@app.route('/knowledge_base')
def knowledge_base():
    """知识库管理页面"""
    entities = get_knowledge_entities()
    # 转换为模板需要的格式
    entity_list = [(e.text, e.label) for e in entities]
    return render_template('knowledge_base.html', entities=entity_list, total=len(entity_list))


# ==================== API接口 ====================

@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """保存标注"""
    data = request.json
    
    new_ann = Annotation(
        file_id=data['file_id'],
        tag_type=data['tag_type'],
        start_index=data['start'],
        end_index=data['end'],
        selected_text=data['text'],
        label=data['label']
    )
    db.session.add(new_ann)
    
    # 更新文件状态
    file = TextFile.query.get(data['file_id'])
    if file:
        file.status = 1
    
    db.session.commit()
    
    # 自动学习：将手动标注添加到知识库
    add_to_knowledge_base(data['text'], data['label'], source='manual')
    
    return jsonify({'status': 'success', 'id': new_ann.id})


@app.route('/api/delete_annotation', methods=['POST'])
def delete_annotation():
    """删除标注"""
    ann_id = request.json.get('id')
    Annotation.query.filter_by(id=ann_id).delete()
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/clear_annotations', methods=['POST'])
def clear_annotations():
    """清空指定文件的所有标注"""
    file_id = request.json.get('file_id')
    Annotation.query.filter_by(file_id=file_id).delete()
    
    # 重置文件状态
    file = TextFile.query.get(file_id)
    if file:
        file.status = 0
    
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/delete_file/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    """删除文件及其所有标注"""
    file = TextFile.query.get_or_404(file_id)
    db.session.delete(file)  # 级联删除会自动删除关联的标注
    db.session.commit()
    return jsonify({'status': 'success', 'message': '文件已删除'})


@app.route('/api/mark_complete/<int:file_id>', methods=['POST'])
def mark_complete(file_id):
    """标记任务为已完成"""
    file = TextFile.query.get_or_404(file_id)
    file.status = 2
    db.session.commit()
    return jsonify({'status': 'success', 'message': '任务已标记为完成'})


@app.route('/api/knowledge_base', methods=['GET'])
def get_knowledge_base_api():
    """获取知识库内容（API）"""
    entities = get_knowledge_entities()
    return jsonify({
        'entities': {e.text: e.label for e in entities},
        'total': len(entities)
    })


@app.route('/api/knowledge_base/entity', methods=['POST'])
def add_knowledge_entity():
    """添加知识库实体"""
    data = request.json
    text = data.get('text', '').strip()
    label = data.get('label', '').strip()
    
    if not text or not label:
        return jsonify({'status': 'error', 'message': '缺少必要参数'}), 400
    
    if add_to_knowledge_base(text, label):
        return jsonify({'status': 'success', 'message': '实体已添加'})
    else:
        return jsonify({'status': 'error', 'message': '实体已存在'}), 400


@app.route('/api/knowledge_base/entity', methods=['DELETE'])
def delete_knowledge_entity():
    """从知识库中删除实体"""
    text = request.json.get('text')
    
    if not text:
        return jsonify({'status': 'error', 'message': '缺少实体文本'}), 400
    
    entity = KnowledgeEntity.query.filter_by(text=text).first()
    if entity:
        db.session.delete(entity)
        db.session.commit()
        return jsonify({'status': 'success', 'message': '实体已删除'})
    else:
        return jsonify({'status': 'error', 'message': '实体不存在'}), 404


@app.route('/api/knowledge_base/clear', methods=['POST'])
def clear_knowledge_base():
    """清空知识库"""
    KnowledgeEntity.query.delete()
    db.session.commit()
    return jsonify({'status': 'success', 'message': '知识库已清空'})


@app.route('/api/knowledge_base/export', methods=['GET'])
def export_knowledge_base():
    """导出知识库为JSON"""
    entities = get_knowledge_entities()
    data = {
        'entities': [
            {
                'text': e.text,
                'label': e.label,
                'source': e.source,
                'created_time': e.created_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            for e in entities
        ],
        'export_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'total': len(entities)
    }
    return jsonify(data)


@app.route('/api/knowledge_base/import', methods=['POST'])
def import_knowledge_base():
    """导入知识库"""
    data = request.json
    entities = data.get('entities', [])
    
    imported_count = 0
    for item in entities:
        if isinstance(item, dict):
            text = item.get('text', '')
            label = item.get('label', '')
        else:
            # 兼容旧格式 {text: label}
            continue
        
        if text and label:
            if add_to_knowledge_base(text, label, source='import'):
                imported_count += 1
    
    return jsonify({
        'status': 'success',
        'message': f'成功导入 {imported_count} 个实体'
    })


# ==================== 启动应用 ====================

if __name__ == '__main__':
    app.run(debug=True, port=5001)