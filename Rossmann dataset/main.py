"""Train the Rossmann sales network and print sample test predictions."""

from pathlib import Path

from model import predict_sales, train_model
from preprocessing import prepare_data


def main():
    data_dir = Path(__file__).resolve().parent
    X_train, X_val, X_test, y_train, y_val = prepare_data(data_dir)
    model = train_model(X_train, y_train, X_val, y_val)
    sales_predictions = predict_sales(model, X_test)
    print("Sample predictions:", sales_predictions[:5])


if __name__ == "__main__":
    main()
