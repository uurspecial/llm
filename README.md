# llm_finetune_project
[PTT](https://docs.google.com/presentation/d/1Xh2swC-2ybWCpR3AK-SQyaFlZVtL4hc4hOL0bh-UOVY/edit?usp=sharing)

## 專案說明
本專案是用來微調 DeBERTa 和 TinyLlama 等大型語言模型，包含資料處理、訓練程式碼與模型結果。

## 資料結構
- `dataset/`：訓練資料（train.csv）
- `main_deberta.py`：DeBERTa 訓練主程式
- `main_tinyllama_lora.py`：TinyLlama LoRA 微調程式
- `main_tinyllama.py`：TinyLlama 微調程式（較大模型）
- `config.py`：共用設定檔
- `utils.py`：資料前處理工具
- `requirements.txt`：依賴套件列表

## 執行方式
1. 建立虛擬環境並安裝依賴：  
python -m venv venv    
source venv/bin/activate   
pip install -r requirements.txt  

2. 執行訓練：
python main_deberta.py和python main_tinyllama_lora.py


## 注意事項
- TinyLlama 1.1B 模型較大，16GB GPU 可能不易跑起來，建議使用較小模型或調整 batch size。
- 模型訓練需較長時間，請確保 GPU 不中斷。

---
