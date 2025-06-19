这个项目是为了参加天池大赛“CCKS2025-大模型生成文本检测”而创建的。
# BERT 文本分类项目

## 安装依赖
```bash
pip install -r requirements.txt
```

## 数据说明
- `data/train.jsonl` 格式：{"text": "...", "label": 0/1}
- `data/test.jsonl` 格式：{"text": "..."}

## 运行
```bash
python main.py
```
