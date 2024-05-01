# Import các thư viện cần thiết
import argparse
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from joblib import dump, load


def preprocess(data, model_dir, save_model=False):
    data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
    transformer_path = os.path.join(model_dir, "transformers/data_filler.joblib")
    if save_model:
        transformer = ColumnTransformer([
            ("mean_imputer", SimpleImputer(strategy='mean'),
             ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age']),
            ("median_imputer", SimpleImputer(strategy='median'), ['Insulin', 'DiabetesPedigreeFunction'])
        ])
        transformer.fit(data)
        dump(transformer, transformer_path)
    else:
        transformer = load(transformer_path)

    data = pd.DataFrame(transformer.transform(data.copy(deep=True)), index=data.index,
                        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Age',
                                 'Insulin', 'DiabetesPedigreeFunction'])

    return data


def feature_engineering(data, model_dir, save_model=False):
    # Feature engineering
    new_feature = pd.cut(data['Glucose'], [0, 120.0, pd.Series.max(data['Glucose'])], right=True,
                         labels=['Normal', 'High'])
    data.insert(len(data.columns), "GlucoseType", value=new_feature)
    data.drop("Glucose", axis="columns", inplace=True)

    # new_feature = pd.cut(data['BMI'], [0, 25, 35, data['BMI'].max()], labels=['Normal', 'ObeseTypeI', 'ObeseTypeII'])
    # data.insert(len(data.columns), "BMIType", value=new_feature)
    # data.drop("BMI", axis="columns", inplace=True)

    new_feature = pd.cut(data['Age'], [0, 25, 35, data['Age'].max()], labels=['young', 'mature', 'senior'])
    data.insert(len(data.columns), "AgeType", value=new_feature)
    data.drop("Age", axis="columns", inplace=True)

    # Encode new feature
    encoder_path = os.path.join(model_dir, "transformers/encoder.joblib")
    if save_model:
        encoder = ColumnTransformer(
            [('glucose_encoder', OrdinalEncoder(), ['GlucoseType']),
             # ('bmi_encoder', OneHotEncoder(), ["BMIType"]),
             ("age_encoder", OneHotEncoder(), ["AgeType"]),
             ]
        )
        encoder.fit(data)

        dump(encoder, encoder_path)
    else:
        encoder = load(encoder_path)

    encoded_features = pd.DataFrame(encoder.transform(data), columns=['GlucoseType',
                                                                      # "BMIType_Normal",
                                                                      # "BMIType_ObeseTypeI", "BMIType_ObeseTypeII",
                                                                      "AgeType_Young", "AgeType_Mature",
                                                                      "AgeType_Senior",
                                                                      ])
    data.drop("GlucoseType", axis="columns", inplace=True)
    # data.drop("BMIType", axis="columns", inplace=True)
    data.drop("AgeType", axis="columns", inplace=True)
    data = pd.concat([data, encoded_features], axis="columns")

    return data


# Hàm để chạy quá trình huấn luyện
def run_train(train_dir, dev_dir, model_dir):
    # Tạo thư mục cho mô hình nếu nó chưa tồn tại
    os.makedirs(model_dir, exist_ok=True)
    temp = os.path.join(model_dir, "transformers")
    os.makedirs(temp, exist_ok=True)

    # Đường dẫn đến các tệp dữ liệu
    train_file = os.path.join(train_dir, 'train.json')
    dev_file = os.path.join(dev_dir, 'dev.json')

    # Đọc dữ liệu huấn luyện và phát triển
    train_data = pd.read_json(train_file, lines=True)
    dev_data = pd.read_json(dev_file, lines=True)

    # Chuẩn bị dữ liệu cho quá trình huấn luyện
    X_train = train_data.drop('Outcome', axis=1)
    Y_train = train_data['Outcome']
    X_dev = dev_data.drop('Outcome', axis=1)
    Y_dev = dev_data['Outcome']

    # Tiền xử lí dữ liệu + feature engineering
    X_train = preprocess(X_train, model_dir, True)
    X_dev = preprocess(X_dev, model_dir, False)
    X_train = feature_engineering(X_train, model_dir, True)
    X_dev = feature_engineering(X_dev, model_dir, False)

    # Tạo và huấn luyện mô hình
    model = LogisticRegression(class_weight='balanced', solver='liblinear')

    search_space = {'C': np.logspace(-4, 4, 20)}
    grid_search = GridSearchCV(estimator=model, param_grid=search_space, scoring='f1')
    grid_search.fit(X_train, Y_train)

    model.set_params(**grid_search.best_params_)

    model.fit(X_train, Y_train)

    predict_path = os.path.join(model_dir, 'diabetes_dev.json')
    y_predict = model.predict(X_dev)
    pd.DataFrame(y_predict, columns=['Outcome']).to_json(predict_path, orient='records', lines=True)

    # Lưu mô hình
    model_path = os.path.join(model_dir, 'trained_model.joblib')
    dump(model, model_path)


# Hàm để chạy quá trình dự đoán
def run_predict(model_dir, input_dir, output_path):
    # Đường dẫn đến mô hình và dữ liệu đầu vào
    model_path = os.path.join(model_dir, 'trained_model.joblib')
    input_file = os.path.join(input_dir, 'test.json')

    # Tải mô hình
    model = load(model_path)

    # Đọc dữ liệu kiểm tra
    test_data = pd.read_json(input_file, lines=True)

    # Chuẩn bị dữ liệu kiểm tra
    X_test = test_data

    # Tiền xử lí dữ liệu + feature engineering
    X_test = preprocess(X_test, model_dir, False)
    X_test = feature_engineering(X_test, model_dir, False)

    # Thực hiện dự đoán
    predictions = model.predict(X_test)

    # Lưu kết quả dự đoán
    pd.DataFrame(predictions, columns=['Outcome']).to_json(output_path, orient='records', lines=True)


# Hàm chính để xử lý lệnh từ dòng lệnh
def main():
    # Tạo một parser cho các lệnh từ dòng lệnh
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Tạo parser cho lệnh 'train'
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--train_dir', type=str)
    parser_train.add_argument('--dev_dir', type=str)
    parser_train.add_argument('--model_dir', type=str)

    # Tạo parser cho lệnh 'predict'
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--model_dir', type=str)
    parser_predict.add_argument('--input_dir', type=str)
    parser_predict.add_argument('--output_path', type=str)

    # Xử lý các đối số nhập vào
    args = parser.parse_args()

    # Chọn hành động dựa trên lệnh
    if args.command == 'train':
        run_train(args.train_dir, args.dev_dir, args.model_dir)
    elif args.command == 'predict':
        run_predict(args.model_dir, args.input_dir, args.output_path)
    else:
        parser.print_help()
        sys.exit(1)


# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    main()
