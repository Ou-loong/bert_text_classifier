
from src.train import train_bert
from src.predict import predict_and_save
def main():
    print("开始训练 BERT 模型...")
    train_bert()

    print("模型训练完成，开始预测...")
    predict_and_save()

if __name__ == "__main__":
    main()
