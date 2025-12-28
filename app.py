import os
import jieba
import jieba.posseg as pseg
from snownlp import SnowNLP
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import re
import json

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
    
    # 关联
    text_annotation = db.relationship('TextAnnotation', backref='file', uselist=False, cascade='all, delete-orphan')
    word_annotations = db.relationship('WordAnnotation', backref='file', lazy=True, cascade='all, delete-orphan')


class TextAnnotation(db.Model):
    """文本整体标注（分类和情感）"""
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('text_file.id'), nullable=False, unique=True)
    text_category = db.Column(db.String(50))  # 文本分类
    text_sentiment = db.Column(db.String(50))  # 文本情感
    sentiment_score = db.Column(db.Float)  # 情感分数
    created_time = db.Column(db.DateTime, default=datetime.utcnow)


class WordAnnotation(db.Model):
    """词语标注（分词+词性+命名实体）"""
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey('text_file.id'), nullable=False)
    word_index = db.Column(db.Integer, nullable=False)
    word = db.Column(db.String(100), nullable=False)
    pos = db.Column(db.String(20))  # 词性
    pos_cn = db.Column(db.String(20))  # 词性中文名
    entity_label = db.Column(db.String(50))  # 命名实体标签


class KnowledgeEntity(db.Model):
    """知识库实体表"""
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), unique=True, nullable=False)
    label = db.Column(db.String(50), nullable=False)
    source = db.Column(db.String(50), default='manual')
    created_time = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


# ==================== 辅助函数 ====================

def get_pos_info(flag):
    """获取词性的中文名和颜色"""
    mapping = {
        'n': ('名词', 'primary'), 'nr': ('人名', 'info'), 'ns': ('地名', 'success'),
        'nt': ('机构', 'danger'), 'nz': ('专名', 'primary'), 'v': ('动词', 'success'),
        'vd': ('副动词', 'success'), 'vn': ('名动词', 'success'), 'a': ('形容词', 'warning'),
        'ad': ('副形词', 'warning'), 'd': ('副词', 'secondary'), 'm': ('数词', 'dark'),
        'q': ('量词', 'dark'), 'r': ('代词', 'secondary'), 'p': ('介词', 'secondary'),
        'c': ('连词', 'secondary'), 'u': ('助词', 'secondary'), 'w': ('标点', 'light'),
        'x': ('标点', 'light'), 't': ('时间', 'purple')
    }
    return mapping.get(flag, ('其他', 'secondary'))


def get_text_category(content):
    """简单的文本分类"""
    keywords = {
        '新闻': ['报道', '记者', '消息', '发布', '宣布', '会议'],
        '科技': ['技术', '科学', '研究', '发明', '创新', '人工智能'],
        '财经': ['股票', '基金', '投资', '经济', '市场', '金融'],
        '体育': ['比赛', '冠军', '球队', '运动员', '足球', '篮球'],
        '娱乐': ['明星', '电影', '演员', '歌手', '综艺', '导演'],
        '教育': ['学校', '学生', '教育', '考试', '老师', '课程'],
    }
    for cat, kws in keywords.items():
        if any(kw in content for kw in kws):
            return cat
    return '其他'


def get_knowledge_dict():
    """获取知识库字典"""
    entities = KnowledgeEntity.query.all()
    return {e.text: e.label for e in entities}


def add_to_knowledge_base(text, label, source='manual'):
    """添加到知识库"""
    existing = KnowledgeEntity.query.filter_by(text=text).first()
    if not existing:
        db.session.add(KnowledgeEntity(text=text, label=label, source=source))
        db.session.commit()
        return True
    return False


# ==================== 页面路由 ====================

@app.route('/')
def index():
    files = TextFile.query.order_by(TextFile.upload_time.desc()).all()
    return render_template('index.html', files=files)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '' or not file:
        return redirect(url_for('index'))
    try:
        content = file.read().decode('utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        content = file.read().decode('gbk')
    db.session.add(TextFile(filename=file.filename, content=content))
    db.session.commit()
    return redirect(url_for('index'))


@app.route('/manual_input', methods=['POST'])
def manual_input():
    task_name = request.form.get('task_name', '').strip()
    text_content = request.form.get('text_content', '').strip()
    if not task_name or not text_content:
        return redirect(url_for('index'))
    if not task_name.endswith('.txt'):
        task_name += '.txt'
    db.session.add(TextFile(filename=task_name, content=text_content))
    db.session.commit()
    return redirect(url_for('index'))


@app.route('/annotate/<int:file_id>')
def annotate(file_id):
    file_data = TextFile.query.get_or_404(file_id)
    word_anns = WordAnnotation.query.filter_by(file_id=file_id).order_by(WordAnnotation.word_index).all()
    text_ann = TextAnnotation.query.filter_by(file_id=file_id).first()
    return render_template('annotate.html', file=file_data, word_anns=word_anns, text_ann=text_ann)


@app.route('/stats')
def stats():
    from sqlalchemy import func
    total_files = TextFile.query.count()
    
    # 统计命名实体标注
    entity_stats = db.session.query(
        WordAnnotation.entity_label, func.count(WordAnnotation.id)
    ).filter(WordAnnotation.entity_label.isnot(None)).group_by(WordAnnotation.entity_label).all()
    
    stats_dict = dict(entity_stats)
    labels = ['人名', '地名', '组织', '时间', '情感']
    complete_stats = [(label, stats_dict.get(label, 0)) for label in labels]
    total_anns = sum(s[1] for s in complete_stats)
    avg_anns = round(total_anns / total_files) if total_files > 0 else 0
    
    return render_template('stats.html',
                           stats_data=complete_stats,
                           labels=labels,
                           counts=[s[1] for s in complete_stats],
                           total_files=total_files,
                           total_anns=total_anns,
                           avg_anns=avg_anns)


@app.route('/knowledge_base')
def knowledge_base():
    entities = KnowledgeEntity.query.order_by(KnowledgeEntity.text).all()
    entity_list = [(e.text, e.label) for e in entities]
    return render_template('knowledge_base.html', entities=entity_list, total=len(entity_list))


# ==================== API接口 ====================

@app.route('/api/smart_annotate/<int:file_id>', methods=['POST'])
def smart_annotate(file_id):
    """智能标注API"""
    file_data = TextFile.query.get_or_404(file_id)
    content = file_data.content
    
    # 清除旧标注
    WordAnnotation.query.filter_by(file_id=file_id).delete()
    TextAnnotation.query.filter_by(file_id=file_id).delete()
    
    # 1. 文本整体标注
    text_category = get_text_category(content)
    try:
        sentiment_score = SnowNLP(content).sentiments
        text_sentiment = '积极' if sentiment_score > 0.6 else ('消极' if sentiment_score < 0.4 else '中性')
    except:
        sentiment_score, text_sentiment = 0.5, '中性'
    
    text_ann = TextAnnotation(
        file_id=file_id, text_category=text_category,
        text_sentiment=text_sentiment, sentiment_score=sentiment_score
    )
    db.session.add(text_ann)
    
    # 2. 分词和词性标注
    knowledge_dict = get_knowledge_dict()
    words = list(pseg.cut(content))
    
    for idx, (word, flag) in enumerate(words):
        pos_cn, _ = get_pos_info(flag)
        entity_label = None
        
        # 知识库匹配
        if word in knowledge_dict:
            entity_label = knowledge_dict[word]
        # 词性判断实体
        elif flag == 'nr':
            entity_label = '人名'
        elif flag == 'ns':
            entity_label = '地名'
        elif flag == 'nt':
            entity_label = '组织'
        elif flag == 't':
            entity_label = '时间'
        
        db.session.add(WordAnnotation(
            file_id=file_id, word_index=idx, word=word,
            pos=flag, pos_cn=pos_cn, entity_label=entity_label
        ))
    
    file_data.status = 1
    db.session.commit()
    
    # 返回标注结果
    word_anns = WordAnnotation.query.filter_by(file_id=file_id).order_by(WordAnnotation.word_index).all()
    return jsonify({
        'status': 'success',
        'text_annotation': {
            'category': text_category,
            'sentiment': text_sentiment,
            'sentiment_score': sentiment_score
        },
        'word_annotations': [{
            'id': w.id, 'index': w.word_index, 'word': w.word,
            'pos': w.pos, 'pos_cn': w.pos_cn, 'entity_label': w.entity_label
        } for w in word_anns]
    })


@app.route('/api/save_all_annotations', methods=['POST'])
def save_all_annotations():
    """保存所有标注"""
    data = request.json
    file_id = data['file_id']
    
    # 更新文本整体标注
    text_ann = TextAnnotation.query.filter_by(file_id=file_id).first()
    if not text_ann:
        text_ann = TextAnnotation(file_id=file_id)
        db.session.add(text_ann)
    text_ann.text_category = data.get('text_category')
    text_ann.text_sentiment = data.get('text_sentiment')
    
    # 更新词语标注
    for w_data in data.get('word_annotations', []):
        word_ann = None
        
        # 优先通过 id 查找
        if w_data.get('id'):
            word_ann = WordAnnotation.query.get(w_data['id'])
        
        # 如果没有 id 或找不到，通过 file_id 和 index 查找
        if not word_ann and 'index' in w_data:
            word_ann = WordAnnotation.query.filter_by(
                file_id=file_id, 
                word_index=w_data['index']
            ).first()
        
        if word_ann:
            if w_data.get('pos'):
                word_ann.pos = w_data.get('pos')
            if w_data.get('pos_cn'):
                word_ann.pos_cn = w_data.get('pos_cn')
            word_ann.entity_label = w_data.get('entity_label')
            # 自动学习到知识库
            if w_data.get('entity_label'):
                add_to_knowledge_base(word_ann.word, w_data['entity_label'], 'manual')
    
    file = TextFile.query.get(file_id)
    if file:
        file.status = 1
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/update_word_annotation', methods=['POST'])
def update_word_annotation():
    """更新单个词语标注"""
    data = request.json
    word_ann = WordAnnotation.query.get(data['id'])
    if word_ann:
        if 'pos' in data:
            word_ann.pos = data['pos']
            word_ann.pos_cn = data.get('pos_cn', '')
        if 'entity_label' in data:
            word_ann.entity_label = data['entity_label']
            if data['entity_label']:
                add_to_knowledge_base(word_ann.word, data['entity_label'], 'manual')
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 404


@app.route('/api/delete_word_entity/<int:word_id>', methods=['DELETE'])
def delete_word_entity(word_id):
    """删除词语的命名实体标注"""
    word_ann = WordAnnotation.query.get(word_id)
    if word_ann:
        word_ann.entity_label = None
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 404


@app.route('/api/update_text_annotation', methods=['POST'])
def update_text_annotation():
    """更新文本整体标注"""
    data = request.json
    file_id = data['file_id']
    text_ann = TextAnnotation.query.filter_by(file_id=file_id).first()
    if not text_ann:
        text_ann = TextAnnotation(file_id=file_id)
        db.session.add(text_ann)
    if 'text_category' in data:
        text_ann.text_category = data['text_category']
    if 'text_sentiment' in data:
        text_ann.text_sentiment = data['text_sentiment']
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/export_annotations/<int:file_id>')
def export_annotations(file_id):
    """导出标注结果"""
    file_data = TextFile.query.get_or_404(file_id)
    text_ann = TextAnnotation.query.filter_by(file_id=file_id).first()
    word_anns = WordAnnotation.query.filter_by(file_id=file_id).order_by(WordAnnotation.word_index).all()
    
    result = {
        'file_info': {
            'id': file_data.id,
            'filename': file_data.filename,
            'content': file_data.content,
            'upload_time': file_data.upload_time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'text_annotation': {
            'category': text_ann.text_category if text_ann else None,
            'sentiment': text_ann.text_sentiment if text_ann else None
        } if text_ann else None,
        'word_annotations': [{
            'index': w.word_index, 'word': w.word,
            'pos': w.pos, 'pos_cn': w.pos_cn, 'entity_label': w.entity_label
        } for w in word_anns],
        'export_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    }
    return jsonify(result)


@app.route('/api/delete_file/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    file = TextFile.query.get_or_404(file_id)
    db.session.delete(file)
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/mark_complete/<int:file_id>', methods=['POST'])
def mark_complete(file_id):
    file = TextFile.query.get_or_404(file_id)
    file.status = 2
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/knowledge_base', methods=['GET'])
def get_knowledge_base_api():
    entities = KnowledgeEntity.query.all()
    return jsonify({'entities': {e.text: e.label for e in entities}, 'total': len(entities)})


@app.route('/api/knowledge_base/entity', methods=['POST'])
def add_knowledge_entity():
    data = request.json
    text, label = data.get('text', '').strip(), data.get('label', '').strip()
    if not text or not label:
        return jsonify({'status': 'error', 'message': '缺少参数'}), 400
    if add_to_knowledge_base(text, label):
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': '已存在'}), 400


@app.route('/api/knowledge_base/entity', methods=['DELETE'])
def delete_knowledge_entity():
    text = request.json.get('text')
    entity = KnowledgeEntity.query.filter_by(text=text).first()
    if entity:
        db.session.delete(entity)
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error'}), 404


@app.route('/api/knowledge_base/clear', methods=['POST'])
def clear_knowledge_base():
    KnowledgeEntity.query.delete()
    db.session.commit()
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True, port=5001)