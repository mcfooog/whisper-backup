# python down.py
# 使用 ModelScope 下载 ct-punc 标点模型（国内直连，速度快）

from modelscope import snapshot_download

# 目标目录
cache_dir = r"C:\whisper\ct-punc"

# ModelScope 上的标点模型 ID
model_id = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"

print(f"开始从 ModelScope 下载标点模型到: {cache_dir}")
model_dir = snapshot_download(model_id, cache_dir=cache_dir)
print(f"✅ 下载完成！模型实际存放路径为: {model_dir}")