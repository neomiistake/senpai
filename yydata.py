import sqlite3
import pandas as pd

# 1. 連接到 SQLite 資料庫（如果不存在則創建）
db = sqlite3.connect('yydata_Tyy.db')  # 資料庫名稱
cursor = db.cursor()  # 創建游標


# # 2. 創建資料表
cursor.execute('''
CREATE TABLE IF NOT EXISTS fight (
    Book_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Face TEXT ,
    Pose TEXT ,
    Age INTEGER ,
    Gender TEXT ,
    Yolo TEXT ,
    Distance FLOAT,
    Total INTEGER
               
);
               ''')

# # 2. 創建資料表
# cursor.execute("""
# ALTER TABLE fight ADD COLUMN YOLO TEXT;
# """)
# # # #pose TEXT NOT NULL 
# # # #face TEXT NOT NULL
# db.commit()  # 提交變更

# # 定義要插入的變數
ID = 1
face = "sad"
pose = "有點危險"
age = 30
genderrr = "女"
YOLO = "people"
DD=1.4
tt = 1430
# Y = "people"

# # path = '1.jpg'

# # # # # # # # # # # INSERT OR REPLACE INTO fight (Book_ID, Title, Author) VALUES (1, '1984', 'George Orwell');

# # 使用變數插入資料
cursor.execute("INSERT OR REPLACE INTO fight (Book_ID,Face,Pose,Age,Gender,Yolo,Distance,Total) VALUES (?,?,?,?,?,?,?,?)", (1000,face,pose,age,genderrr,YOLO,DD,tt))
# # # # #保有原有數據。更新
# # # # cursor.execute("UPDATE fight SET Total = ? WHERE Book_id = ?", (DD,1))
# try:
#     # 建立游標（cursor）
#     cursor = db.cursor()

#     # 刪除所有資料
#     delete_query = "DELETE FROM fight;"
#     cursor.execute(delete_query)

#     # 確認刪除
#     db.commit()
#     print("資料已成功刪除！")

# except sqlite3.Error as e:
#     print(f"發生錯誤: {e}")
#     db.rollback()  # 發生錯誤時回滾

db.commit()  # 提交變更

# # ID = ID+1

# # # # 刪除表
# # # cursor.execute("DROP TABLE IF EXISTS fight")
# # # db.commit()

# 查詢資料的副程式
def run_query(query):
    return pd.read_sql_query(query, db)

# 查詢所有書籍
query = "SELECT * FROM fight;"
fight_df = run_query(query)

# # 將 DataFrame 寫入 Excel 檔案
# excel_file = 'fight_T.xlsx'
# fight_df.to_excel(excel_file, index=False)  # index=False 表示不寫入行索引


# 顯示結果
print(fight_df)

# # # 清空名為 'datatest' 的表中的所有數據
# # cursor.execute("DELETE FROM fight")
# # db.commit()



# ##抓取資料庫資料
# # 分別存放每個欄位的列表
# # ID = []
# # Face = []
# # Pose = []
# # Age = []
# # Gender = []
# # Yolo = []
# # Distance = []

# # # 將每個欄位的數據分別放入不同的列表
# # for row in fight_df.itertuples(index=False):
# #     # if len(row) >=7:
# #         ID.append(row.Book_ID)
# #         Face.append(row[1])     # face 欄位
# #         Pose.append(row[2])     # pose 欄位
# #         Age.append(row[3])      # age 欄位
# #         Gender.append(row[4])
# #         Yolo.append(row[5])
# #         Distance.append(row[6])
# #     # else:
# #         # print("跳過不完整的行:", row)

# # # 印出每個欄位的列表
# # print("ID:", ID)
# # print("face:", Face)
# # print("pose:", Pose)
# # print("age:", Age)
# # print("gender:", Gender)
# # print("yolo:", Yolo)
# # print("distancd:", Distance)



# 5. 關閉資料庫連接
db.close()






#回傳樹梅派

# import sqlite3
# import requests
# import time

# # 資料庫和伺服器設定
# db_path = "yydata_T.db"  # 資料庫檔案路徑
# table_name = "fight"  # 資料表名稱
# raspberry_pi_url = "http://192.168.0.9:5000/update"  # 樹莓派的接收 API URL
# # 連接資料庫
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # db = sqlite3.connect('yydata_T.db')  # 資料庫名稱
# # cursor = db.cursor()  # 創建游標


# # 查詢 s_total 欄位的所有值
# cursor.execute(f"SELECT Total FROM {table_name}")
# Total_values = cursor.fetchall()

# # 依序將 s_total 欄位的值傳送給樹莓派
# for Total in Total_values:
#     # 將 s_total 從 tuple 提取出來並格式化成 JSON
#     data = {"s_total": Total[0]}
    
#     # 發送 HTTP POST 請求
#     response = requests.post(raspberry_pi_url, json=data)
    
#     # 檢查回應狀態
#     if response.status_code == 200:
#         print(f"成功傳送 s_total: {Total[0]}")
#     else:
#         print(f"傳送失敗 s_total: {Total[0]}, 錯誤狀態碼: {response.status_code}")
    
#     # 避免過快傳送，設置延遲
#     time.sleep(1)  # 每次傳送之間延遲1秒

# # 關閉資料庫連接
# conn.close()
