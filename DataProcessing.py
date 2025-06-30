# This script downloads a file from a given URL and saves it to a specified local path.
from urllib.request import urlretrieve

# Define the URL of the file to be downloaded
medical_changes_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'

# Download the file and save it as 'medical-charges.csv' if it already exists, overwriting it
urlretrieve(medical_changes_url, 'medical.csv')

# run the following command to install the pandas library to read, write, and manipulate the CSV file
# pip install pandas --quiet

# Import the pandas library to handle the CSV file
# A CSV file is a tabular file
import pandas as pd

# Read the downloaded CSV file into a pandas DataFrame
medical_df = pd.read_csv('medical.csv')

# print the entire contents of the DataFrame medical_df to the screen
#print(medical_df)

#print general information about the medical_df DataFrame
#print(medical_df.info())

#print general statistics for the numeric columns in the DataFrame
#print(medical_df.describe())

#matplotlib and seaborn are libraries for data visualization
#matplotlib is used for creating static, animated, and interactive visualizations in Python
#seaborn is built on top of matplotlib and provides a high-level interface for drawing attractive statistical graphics
#pip install matplotlib seaborn --quiet

# Plotly Express is a library used for creating interactive charts
import plotly.express as px

#import the matplotlib library, a library used for plotting charts in Python
import matplotlib
import matplotlib.pyplot as plt

#Seaborn helps create beautiful and easy-to-use statistical charts in Python
import seaborn as sns

# This line is used in Jupyter notebooks to display plots inline
#%matplotlib inline 

#set the style of seaborn plots to 'darkgrid'
sns.set_style('darkgrid')

#set the default font size for all plots created with matplotlib (and seaborn) to 14
matplotlib.rcParams['font.size'] = 14

#set the default size for all plots created with matplotlib (and seaborn) to 10 inches wide and 6 inches tall
matplotlib.rcParams['figure.figsize'] = (10, 6)

#set the default background color for all matplotlib (and seaborn) plots to transparent
# The chart will have no background color, which is useful when you want to insert the chart into different backgrounds without it being obscured
matplotlib.rcParams['figure.facecolor'] = '#00000000'

#print descriptive statistics for the 'age' column in the medical_df DataFrame
#print(medical_df.age.describe())

'''
#This code uses Plotly Express to create a histogram for the 'age' column in the medical_df DataFrame
fig = px.histogram( # frequency chart
    medical_df, 
    x='age', # The X axis will represent the age values 
    marginal='box', # add a small boxplot (box chart) above or beside the histogram
    nbins=47, # specify the number of columns (bins) in the histogram
    color='smoker', #color the columns in the chart based on the value of the 'smoker' column in the medical_df DataFrame
    color_discrete_sequence=['red','grey'], #specify the color of the columns in the histogram chart
    title = 'describe the age colum'
)

# Show the histogram plot
fig.update_layout(bargap = 0.1) # set the gap between the bars
fig.show()
'''

'''
fig = px.scatter( # scatter plot
    medical_df,
    x='age',
    y= 'charges',
    color='smoker',
    title='Charges vs Age',
    opacity = 0.8, # set the opacity of the points (0 is fully transparent, 1 is fully opaque)
    hover_data=['sex'] # when hovering over a point, additional information about gender (sex) will be displayed
)
fig.update_traces(marker_size=5) # set the marker size in the scatter plot to 5, making the points easier to see
fig.show()
'''


'''
fig = px.violin( # violin plot
    medical_df,
    x='smoker',
    y='charges'
)
fig.show()
'''

'''
sns.barplot( # bar chart
    data = medical_df,
    x='smoker',
    y='charges',
)

plt.title('smoker and charges')
plt.show()
'''

'''
# Used to calculate the correlation coefficient between two numeric columns in the DataFrame
print(medical_df['charges'].corr(medical_df['age']))
print(medical_df.charges.corr(medical_df.age))

# Close to 1: Strong positive correlation (both increase or both decrease)
# Close to -1: Strong negative correlation (one increases, the other decreases)
# Close to 0: No linear correlation

'''

'''
smoker_values = {'no':0,'yes':1} # This code maps the 'smoker' column values to numeric values (0 for 'no', 1 for 'yes')
smoker_numeric = medical_df.smoker.map(smoker_values)  
print(smoker_numeric)

print(medical_df.charges.corr(smoker_numeric))
'''

#print(medical_df[['charges', 'age', 'bmi', 'children']].corr())

'''
smoker_values = {'no':0,'yes':1} # This code maps the 'smoker' column values to numeric values (0 for 'no', 1 for 'yes')
smoker_numeric = medical_df.smoker.map(smoker_values)  
medical_df['smoker_numeric']= smoker_numeric # Add the numeric smoker column to the DataFrame
print(medical_df.smoker_numeric)

medical_df[['age','bmi','children','smoker_numeric']].corr()
'''

'''
sns.heatmap(     # Draw a heatmap showing the correlation matrix between the numeric columns (age, bmi, children, charges) in the DataFrame
    medical_df[['age','bmi','children','charges']].corr(),
    cmap='Reds', # Use the red color palette for the heatmap
    annot=True # Display the numeric value of the correlation coefficient on each cell
)
plt.title('correlation heatmap')
plt.show()
'''

'''
# Filter the DataFrame to include only non-smokers
non_smoker_df = medical_df[medical_df.smoker == 'no']  
sns.scatterplot( # Create a scatter plot for non-smokers
    data = non_smoker_df,
    x='age',
    y='charges',
    alpha = 0.7, # Set the transparency of the points (0 is fully transparent, 1 is fully opaque)
    s = 15 # Set the size of the points in the scatter plot
)
plt.title('Non-smokers: Age vs Charges')
'''


# simple linear regression model
# estimate medical charges based on age, slope (w), and intercept (b)
def estimate_charges(age,w,b):
    return w * age + b  

'''
# Example usage of the estimate_charges function
w=50
b=100
print(estimate_charges(30,w,b)) 
#output: the estimated cost at age 30 is 1600, which is much lower than the actual cost in the medical_df data
'''

# try with the data of non_smoker_df
non_smoker_df = medical_df[medical_df.smoker == 'no']
ages = non_smoker_df.age
#print(ages)

w=50
b=100
estimated_charges = estimate_charges(ages, w, b) 
#print(estimated_charges)
#print(non_smoker_df.charges)
# The estimated charges at different ages for non-smokers are very different from the actual charges in the non_smoker_df data

'''
# Line plot between age (ages) and estimated charges (estimate_charges) for the non-smoker group
plt.plot(
    ages,
    estimated_charges
)
plt.xlabel('Age')
plt.ylabel('Estimated Charges')
plt.title('Estimated Charges vs Age for Non-Smokers')
plt.show()
'''

'''
# compare the estimated values with the actual values using a line plot and a scatter plot
target = non_smoker_df.charges
#print(target)

plt.plot(
    ages,
    estimated_charges,
    'r',
    alpha=0.9
)

plt.scatter(
    ages,
    target,
    alpha = 0.8,
    s =8
)
plt.xlabel('Age')
plt.ylabel('Charges')

# add a legend to the chart to distinguish between the two types of data
# 'Estimate': The line representing estimated charges (from the linear regression function, plotted using plt.plot)
# 'Actual': The actual data points (plotted using plt.scatter)
plt.legend(['Estimate','Actual']) 
plt.show()
'''

'''
def try_parameter(w,b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    
    estimated_charges = estimate_charges(ages, w, b)  

    plt.plot(ages, estimated_charges, 'r', alpha=0.9)  
    plt.scatter(ages, target, alpha=0.8, s=8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual'])
    plt.show()

    loss = rmse(target, estimated_charges)
    print('rmse:',loss)
'''


# để sửa chữa điều này, chúng ta cần sử dụng hồi quy tuyến tính để tìm ra giá trị tốt nhất cho w và b
#hãy thử điều chỉnh độ dốc và độ lệch nếu muốn đi lên,xuống hãy điều b và muốn tăng, giảm độ dốc hãy điều chỉnh w
#try_parameter(400, 5000)  # Example usage with w=400 and b=5000
#try_parameter(400, 1000) # độ lệch đã đi xuống đáng kể, nhưng vẫn chưa đủ tốt
#try_parameter(400, -6000) # độ lệch đã đi xuống đáng kể, độ dốc vẫn chưa tốt
#try_parameter(310, -6000) # cứ thử đi thử lại với các giá trị khác nhau của w và b

# để tìm ra giá trị tốt nhất cho w và b, chúng ta cần sử dụng hồi quy tuyến tính
import numpy as np

#hàm rmse: đánh giá độ chính xác của mô hình dự đoán
#RMSE càng nhỏ, mô hình dự đoán càng sát với dữ liệu thực tế.
#RMSE lớn nghĩa là dự đoán còn sai lệch nhiều.
def rmse (target, predictions):
    return np.sqrt(np.mean((target - predictions) ** 2)) 

'''
w=50
b=100
#try_parameter(w, b)  # Initial try with w=50 and b=100
target  = non_smoker_df.charges # chứa chi phí thực tế của nhóm không hút thuốc
predicted = estimate_charges(non_smoker_df.age, w, b)  # Tính giá trị dự đoán predicted bằng hàm estimate_charges với các giá trị w, b vừa chọn
#print(estimate_charges(non_smoker_df.age, w, b))
print(rmse(target, predicted))  # in ra giá trị RMSE giữa dự đoán và thực tế để đánh giá mức độ chính xác của mô hình với bộ tham số này
'''


def try_parameter(w,b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    
    estimated_charges = estimate_charges(ages, w, b)  

    plt.plot(ages, estimated_charges, 'r', alpha=0.9)  
    plt.scatter(ages, target, alpha=0.8, s=8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Estimate', 'Actual'])
    plt.show()

    loss = rmse(target, estimated_charges)
    print('rmse:',loss)

#try_parameter(50, 100) #output: rmse: 8461.949562575493
#try_parameter(50, 300) #output: rmse: 8312.556156041677
#try_parameter(350, 100) #output: rmse: 7266.699605113179
#try_parameter(350, -4000) #output: rmse: 4991.993804156943


# dùng LinearRegression để tự động tìm giá trị tốt nhất cho w (hệ số góc) và b (hệ số chệch) trong mô hình hồi quy tuyến tính, giúp dự đoán chi phí (charges) dựa trên tuổi (age) cho nhóm không hút thuốc
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # Tạo một mô hình hồi quy tuyến tính mới

# tạo ma trận 2 chiều cho đầu vào (input) của mô hình
inputs = non_smoker_df[['age']]  # Chọn cột 'age' làm đầu vào (input) cho mô hình
# tạo ma trận 1 chiều cho đầu ra (target) của mô hình
target = non_smoker_df['charges']  # Chọn cột 'charges' làm đầu ra (target) cho mô hình
#print('inputs.shape: ',inputs.shape)  # In ra kích thước của ma trận đầu vào (sẽ là (n, 1) với n là số lượng mẫu trong đầu vào)
#print('target.shape: ',target.shape)  # In ra kích thước của ma trận đầu ra (sẽ là (n,) với n là số lượng mẫu trong đầu ra)

# Huấn luyện mô hình với dữ liệu đầu vào và đầu ra
#model.fit(X,Y) là dạng chuẩn với X là ma trận 2 chiều đầu vào (DataFrame hoặc mảng numpy) và Y là ma trận 1 chiều đầu ra (Series hoặc mảng numpy)
#kiểm tra bằng cách:
#print(type(inputs))  #output: <class 'pandas.core.frame.DataFrame'>
#print(type(target))  #output: <class 'pandas.core.series.Series'>

# Huấn luyện mô hình với dữ liệu đầu vào và đầu ra
model.fit(inputs, target)  # chỉ cần gọi như vậy là đã huấn luyện mô hình với dữ liệu đầu vào và đầu ra. giờ chỉ việc lấy w và b ra sử dụng

'''
# kiểm tra kết quả của mô hình hồi quy tuyến tính
print('w:', model.coef_[0]) # In ra giá trị hệ số góc (w) của mô hình hồi quy tuyến tính
print('b:', model.intercept_) # In ra giá trị hệ số chệch (b) của mô hình hồi quy tuyến tính
try_parameter(model.coef_[0], model.intercept_)
'''


'''
# Dự đoán chi phí (charges) cho nhóm không hút thuốc dựa trên tuổi (age) bằng mô hình hồi quy tuyến tính đã huấn luyện
#model.predict(X) là dạng chuẩn với X là ma trận 2 chiều đầu vào (DataFrame hoặc mảng numpy)
print(model.predict(np.array(
    [[23],[37],[61]]
)))
# báo lỗi warnings.warn( thì kệ vì nó không ảnh hưởng gì đến kết quả dự đoán, chỉ là cảnh báo thôi

# Vẽ biểu đồ phân tán (scatter plot) giữa tuổi (age) và chi phí thực tế (charges) cho nhóm không hút thuốc để so sánh với đường hồi quy tuyến tính
plt.plot(
    non_smoker_df.age, 
    model.predict(non_smoker_df[['age']]), # Dự đoán chi phí (charges) cho nhóm không hút thuốc dựa trên tuổi (age) bằng mô hình hồi quy tuyến tính đã huấn luyện
    'r', # Màu đỏ cho đường hồi quy tuyến tính
    alpha=0.9 # Độ trong suốt của đường hồi quy tuyến tính
)
sns.scatterplot(
    data=non_smoker_df,
    x='age',
    y='charges',
    alpha = 0.7, # Set the transparency of the points (0 is fully transparent, 1 is fully opaque)
    s = 15 # Set the size of the points in the scatter plot
)
plt.show()
'''

'''
predictions = model.predict(inputs)  # Dự đoán chi phí (charges) cho nhóm không hút thuốc dựa trên tuổi (age) bằng mô hình hồi quy tuyến tính đã huấn luyện trả về mảng 1 chiều
#print('predictions:', predictions)  # In ra kết quả là mảng 1 chiều dự đoán chi phí (charges) cho nhóm không hút thuốc dựa trên tuổi (age) bằng mô hình hồi quy tuyến tính đã huấn luyện 
#lưu ý có thể sẽ có có một vài trường hợp khác vì đây là dự đoán, không phải là giá trị thực tế và sẽ có một vài người có cùng độ tuổi nhưng chi phí khác nhau nên chỉ lấy được giá trị trung bình của chi phí cho mỗi độ tuổi

print(rmse(target, predictions))  # Tính toán và in ra giá trị RMSE giữa chi phí thực tế (target) và chi phí dự đoán (predictions) để đánh giá độ chính xác của mô hình hồi quy tuyến tính
#output: 4662.505766636395 có thể cho thấy mô hình hồi quy tuyến tính này đã dự đoán khá chính xác chi phí y tế cho nhóm không hút thuốc dựa trên tuổi của họ với mức chênh lệch trung bình khoảng 4662.505766636395 nhưng nhìn vào biểu đồ thì có thể thấy mức chênh lệch ít hơn trừ 1 vài trường hợp ngoại lệ
# lưu ý mức 4000 là không tệ
'''

# w được lưu trữ trong model.coef_
#print(model.coef_) # cho biết mỗi năm tuổi tăng thì chi phí dự đoán tăng bao nhiêu
# b được lưu trữ trong model.intercept_
#print(model.intercept_) # chi phí dự đoán khi tuổi bằng 0
'''
try_parameter(model.coef_, model.intercept_)  # Sử dụng hàm try_parameter để vẽ biểu đồ với hệ số góc (w) và hệ số chệch (b) đã được huấn luyện từ mô hình hồi quy tuyến tính
'''

'''
from sklearn.linear_model import SGDRegressor

# Huấn luyện mô hình LinearRegression
model = LinearRegression()
model.fit(inputs, target)
w_lr = model.coef_[0]
b_lr = model.intercept_
predictions_lr = model.predict(inputs)
rmse_lr = rmse(target, predictions_lr)

# Huấn luyện mô hình SGDRegressor
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(inputs, target)
w_sgd = sgd_model.coef_[0]
b_sgd = sgd_model.intercept_[0]
predictions_sgd = sgd_model.predict(inputs)
rmse_sgd = rmse(target, predictions_sgd)

# In kết quả so sánh
print("LinearRegression: w =", w_lr, ", b =", b_lr, ", RMSE =", rmse_lr)
print("SGDRegressor:    w =", w_sgd, ", b =", b_sgd, ", RMSE =", rmse_sgd)

# Vẽ biểu đồ so sánh
plt.scatter(inputs, target, alpha=0.7, s=15, label='Actual')
plt.plot(inputs, predictions_lr, 'r', alpha=0.9, label='LinearRegression')
plt.plot(inputs, predictions_sgd, 'g--', alpha=0.9, label='SGDRegressor')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()
# Kết quả so sánh giữa LinearRegression và SGDRegressor
# LinearRegression thường cho kết quả RMSE nhỏ hơn hoặc bằng SGDRegressor (vì nó tìm nghiệm tối ưu tuyệt đối).
# SGDRegressor có thể cho kết quả gần đúng, nhanh hơn với dữ liệu lớn, nhưng RMSE có thể lớn hơn một chút.
'''

'''
# thử với trường hợp người hút thuốc
smoker_df = medical_df[medical_df.smoker == 'yes'] # Lọc dữ liệu để chỉ lấy những người hút thuốc
inputs = smoker_df[['age']] # Chọn cột 'age' làm đầu vào (input) cho mô hình
target = smoker_df['charges'] # Chọn cột 'charges' làm đầu ra (target) cho mô hình
model = LinearRegression() # Tạo một mô hình hồi quy tuyến tính mới
model.fit(inputs,target) # Huấn luyện mô hình với dữ liệu đầu vào và đầu ra
w = model.coef_
b = model.intercept_
print('RMSE: ',rmse(target, model.predict(inputs))) # Tính toán và in ra giá trị RMSE giữa chi phí thực tế (target) và chi phí dự đoán (predictions) để đánh giá độ chính xác của mô hình hồi quy tuyến tính
predictions = model.predict(np.array(
    [[23],[37],[61]]
)) # Dự đoán chi phí (charges) cho nhóm hút thuốc dựa trên tuổi (age) bằng mô hình hồi quy tuyến tính đã huấn luyện

print(predictions) # In ra kết quả dự đoán chi phí (charges) cho nhóm hút thuốc dựa trên tuổi (age) bằng mô hình hồi quy tuyến tính đã huấn luyện

sns.scatterplot(
    data=smoker_df,
    x='age',
    y='charges',
)
plt.plot(
    smoker_df.age,
    model.predict(inputs), # Dự đoán chi phí (charges) cho nhóm hút thuốc dựa trên tuổi (age) bằng mô hình hồi quy tuyến tính đã huấn luyện
    'r'
)
plt.show()
'''

'''
#B1: lọc dữ liệu cần chọn
non_smoker_df = medical_df[medical_df.smoker == 'no']

#B2: tạo inputs và target
inputs = non_smoker_df[['age']]
target = non_smoker_df['charges']

#B3: tạo và huấn luyện mô hình
model = LinearRegression().fit(inputs,target)

#B4: tạo dự đoán (predictions)
predictions = model.predict(inputs)

#B5: Tính toán độ sai số (loss) để đánh giá mô hình
loss = rmse(target, predictions)
print('RMSE:', loss)  

# B6: Vẽ biểu đồ so sánh giữa dự đoán và thực tế
sns.scatterplot(
    data = non_smoker_df,
    x = 'age',
    y = 'charges'
)
plt.plot(
    non_smoker_df.age,
    model.predict(inputs),
    'r'
)
plt.show()
'''

'''
non_smoker_df = medical_df[medical_df.smoker == 'no']

inputs = non_smoker_df[['age','bmi']]
target = non_smoker_df['charges']

model = LinearRegression().fit(inputs, target)

predictions = model.predict(inputs)
loss = rmse(target, predictions)
print('LOSS:', loss)

print('[w1 w2] b :',model.coef_, model.intercept_)

print(non_smoker_df[['age','bmi','charges']].corr())
'''
'''
fig = px.strip(
    non_smoker_df,
    x='children',
    y='charges'
)
fig.show()
'''

'''
non_smoker_df = medical_df[medical_df.smoker == 'no']
inputs = non_smoker_df[['age','bmi','children']]
targer = non_smoker_df['charges']

model =LinearRegression().fit(inputs, targer)
predictions =model.predict(inputs)

loss = rmse(targer, predictions)

print('LOSS:', loss)

sns.scatterplot(
    data = non_smoker_df,
    x='age',
    y='charges'
)
plt.plot(
    non_smoker_df.age,
    model.predict(inputs),
    'r'
)
plt.show()
'''

'''
inputs = medical_df[['age','bmi','children','sex','smoker','region']]
inputs = pd.get_dummies(inputs, drop_first=True)  # Chuyển tất cả các biến phân loại trong DataFrame inputs thành các cột số (0/1) bằng phương pháp one-hot encoding.
#Tham số drop_first=True sẽ bỏ đi một giá trị đầu tiên của mỗi biến phân loại để tránh dư thừa (giảm đa cộng tuyến)
print(inputs.head())

target = medical_df['charges']

model = LinearRegression().fit(inputs, target)

predictions = model.predict(inputs)
loss = rmse(target, predictions)

print('LOSS:', loss)



sns.scatterplot(
    data = medical_df,
    x='age',
    y='charges'
)
plt.plot(
    medical_df.age,
    predictions,
    'r'
)
plt.show()

# inputs1 = medical_df[['charges','age','bmi','children','sex','smoker','region']]
# inputs1 = pd.get_dummies(inputs1, drop_first=True) 
# print(inputs1.corr())
'''

'''
# Chuyển đổi DataFrame chỉ gồm các cột số, trong đó các biến phân loại đã được mã hóa thành các cột 0/1
# C1: sử dụng pd.get_dummies
medical_df = pd.get_dummies(medical_df,drop_first=True)
print(medical_df)

# C2: Dùng .map(smoker_codes)
smoker_codes = {'no':0,'yes':1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)
print(medical_df)
'''

'''#VD: sử dụng C2: Dùng .map(smoker_codes)
smoker_codes = {'no':0,'yes':1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)


inputs = medical_df[['age','bmi','smoker_code','children']]
target = medical_df['charges']

model = LinearRegression().fit(inputs,target)

predictions = model.predict(inputs)

loss = rmse(target,predictions)

print('LOSS: ',loss)

sns.scatterplot(
    data = medical_df,
    x= 'age',
    y='charges'
)
plt.plot(
    medical_df.age,
    predictions,
    'r'
)
plt.show()
'''

'''
#VD sử dụng cách 1: pd.get_dummies
medical_df = pd.get_dummies(medical_df,drop_first=True)

inputs = medical_df[['age','bmi','smoker_yes','children']]
target = medical_df['charges']

model = LinearRegression().fit(inputs,target)

predictions = model.predict(inputs)

loss = rmse(target,predictions)

print('LOSS: ',loss)

sns.scatterplot(
    data = medical_df,
    x= 'age',
    y='charges'
)
plt.plot(
    medical_df.age,
    predictions,
    'r'
)
plt.show()
'''

'''
smoker_codes = {'no':0,'yes':1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

sex_codes = {'female':0,'male':1}
medical_df['sex_code'] = medical_df.sex.map(sex_codes)

inputs = medical_df[['age','bmi','smoker_code','children','sex_code']]
target = medical_df['charges']

model = LinearRegression().fit(inputs,target)

predictions = model.predict(inputs)

loss = rmse(target,predictions)

print('LOSS: ',loss)

sns.scatterplot(
    data = medical_df,
    x= 'age',
    y='charges'
)
plt.plot(
    medical_df.age,
    predictions,
    'r'
)
plt.show()
'''




#B1: thu thập dữ liệu
#print(medical_df)

#B2: Tiền xử lý dữ liệu
smoker_codes = {'no':0,'yes':1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

non_smoker_df = medical_df[medical_df.smoker_code == 0]
#print(non_smoker_df)


# sns.scatterplot(
#     data = non_smoker_df,
#     x='age',
#     y='charges'
# )
#try_parameter(500,1000)
#plt.show()


inputs = non_smoker_df[['age']]
target = non_smoker_df['charges']

# xây dụng mô hình bằng skit-learn
model = LinearRegression().fit(inputs,target)

predictions = model.predict(inputs)

#đánh giá mô hình 
Loss = rmse(target,predictions)

print('LOSS: ',Loss)

#vẽ đồ thị
sns.scatterplot(
    data = non_smoker_df,
    x='age',
    y='charges'
)
plt.plot(
    non_smoker_df.age,
    predictions,
    'r'
)
plt.show()