import pandas as pd

def load_data():
    data_train = pd.read_csv('files/train.csv', encoding="ISO-8859-1")
    data_test = pd.read_csv('files/test.csv', encoding="ISO-8859-1")
    data_product_desc = pd.read_csv('files/product_descriptions.csv', encoding="ISO-8859-1")
    data_attr = pd.read_csv('files/attributes.csv', encoding="ISO-8859-1")
    data_brand = data_attr[data_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    data_b1 = data_attr[data_attr.name == "Bullet01"][["product_uid", "value"]].rename(columns={"value": "bullet1"})
    data_b2 = data_attr[data_attr.name == "Bullet02"][["product_uid", "value"]].rename(columns={"value": "bullet2"})
    data_b3 = data_attr[data_attr.name == "Bullet03"][["product_uid", "value"]].rename(columns={"value": "bullet3"})
    data_b4 = data_attr[data_attr.name == "Bullet04"][["product_uid", "value"]].rename(columns={"value": "bullet4"})
    data_material = data_attr[data_attr.name == "Material"][["product_uid", "value"]].rename(columns={"value": "material"})
    train_count = data_train.shape[0]
    data = pd.concat((data_train, data_test), axis=0, ignore_index=True)
    data = pd.merge(data, data_product_desc, how='left', on='product_uid')
    data = pd.merge(data, data_brand, how='left', on='product_uid')
    data = pd.merge(data, data_b1, how='left', on='product_uid')
    data = pd.merge(data, data_b2, how='left', on='product_uid')
    data = pd.merge(data, data_b3, how='left', on='product_uid')
    data = pd.merge(data, data_b4, how='left', on='product_uid')
    data = pd.merge(data, data_material, how='left', on='product_uid')

    return train_count, data

load_data()
#print load_data()[1]['product_description']