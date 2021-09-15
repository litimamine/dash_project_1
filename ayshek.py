# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 00:46:35 2021

@author: DELL
"""

# Commented out IPython magic to ensure Python compatibility.
# to be able to work with tabular and numerical data
import pandas as pd
import numpy as np

#Vizualization (only if necessary, because we are using plotly instead)
import matplotlib.pyplot as plt


#Interactive Graphing 
#import plotly

#import chart_studio.plotly as py ( necessary instances for plotly , like plotly express)
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go


# to make subplots in plotly
from plotly.subplots import make_subplots 


# Use Plotly on the localhost
cf.go_offline()



# Loading the data into the data frame ( five dataframes in total, extracted from google analytics and Woocommerce of the company)
data = pd.read_csv('https://raw.githubusercontent.com/litimamine/dash_project_1/main/stats.csv', error_bad_lines=False)
data2 = pd.read_csv('https://raw.githubusercontent.com/litimamine/dash_project_1/main/customers.csv', error_bad_lines=False)
data3 = pd.read_csv('https://raw.githubusercontent.com/litimamine/dash_project_1/main/revenues.csv', error_bad_lines=False)
data4 = pd.read_csv('https://raw.githubusercontent.com/litimamine/dash_project_1/main/daily_active_users.csv', error_bad_lines=False)
data5 = pd.read_csv('https://raw.githubusercontent.com/litimamine/dash_project_1/main/average_daily_timespent.csv', error_bad_lines=False)
# setting the column "country" as an index
# inplace param: filling the space caused by column shift by the other columns 
data.set_index('Country' , inplace = True)

data

"""### Since this dataset has already been treated with Excel, we move to the vizualisation part:

### --- Interactive Vizualization
"""

# Figure 1

# creating a figure object and setting up the title
fig = go.Figure(layout=dict(title=dict(text="PLATFORM NEW VISITORS AMONG TOTAL VISITORS")))

#setting the X(counry names) and Y(frequency of visits), and the colour for the colum nbr_visitors
fig.add_trace(go.Bar(
    x= data.index,
    y= data.nbr_visitors,
    name='total visitors',
    marker_color='indianred'
))

#setting the (counry names) and Y(frequency of visits), and a diffirent colour colour for the colum new_visitors
fig.add_trace(go.Bar(
    x= data.index,
    y=data.new_visitors,
    name='new visitors',
    marker_color='lightsalmon'
))

# Here we update the layout through fixing different params, we also modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(barmode='group', xaxis_tickangle=-45, plot_bgcolor='white', 
                  autosize=False, width=900, height=500 , 
                  title_font={'size':22}, title_x=0.5 )


# the 'Sessions' column
data.Sessions

#Sorting the sessions column for visualization purposes (so that it looks better)
sorted_sessions = data.Sessions.sort_values(ascending= True)

# figure 2

# creating a figure object and setting up the title and the bar plot orientation (h = horisontal bar plot)
fig2 = px.bar(sorted_sessions,  title='NUMBER OF WEB/MOBILE SESSIONS ACROSS COUNTRIES', orientation='h')

# updating the layout with different parameters
fig2.update_layout(title_font={'size':22}, title_x=0.5 , plot_bgcolor='white',showlegend=False,
                   xaxis_title_text=' <b> Sessions in thousands </b>', yaxis_title_text=' <b> Countries </b> ', 
                   autosize=False, width=900, height=500 )
# updating the trace colour
fig2.update_traces(marker_color=  px.colors.sequential.Aggrnyl)


# rebound rate column 
# as shown in the output, the % sign is attached to values, which makes them taken by python as a string type
data['rebound_rate']

# Creating a function that turn the rebound_rate from string datatype to numeric to be able to vizualize it
for i in range (len(data)): 
  data['rebound_rate'][i] = int(''.join(filter(str.isnumeric, data.rebound_rate[i]))) / 100

# new datatype of the column
data['rebound_rate']

#figure 3

labels =  ["Rebound rate (%)","1 - rebound Rate (%)"]

# Create subplots: use 'domain' type for Pie subplots (with a matrix of 3X3 subplots)
fig3 = make_subplots(rows=3, cols=3, specs= [[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}],
                                                   [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}],
                                                   [{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]] ) 

#Adding the plot (donut chart) corresponding to each subplot 
#data.rebound_rate[0] is the value corresponding to the first line in the column
#100 - data.rebound_rate[0] has as a goal highlighting the empty area necessary to accomplish 100% rate 
#  1, 1 refers to the first row-first column subplot in the grid

fig3.add_trace(go.Pie(labels = labels, values=[data.rebound_rate[0], 100 - data.rebound_rate[0] ], 
                            name= data.rebound_rate.index[0] ) ,
                     1, 1)
fig3.add_trace(go.Pie(labels = labels, values=[data.rebound_rate[1], 100 - data.rebound_rate[1] ], 
                            name= data.rebound_rate.index[1]),
                     1, 2)
fig3.add_trace(go.Pie(labels = labels, values=[data.rebound_rate[2], 100 - data.rebound_rate[2] ], 
                            name =data.rebound_rate.index[2]),
                     1, 3)
fig3.add_trace(go.Pie(labels = labels, values=[data.rebound_rate[3], 100 - data.rebound_rate[3] ], 
                            name= data.rebound_rate.index[3]),
                     2, 1)
fig3.add_trace(go.Pie(labels = labels, values=[data.rebound_rate[4], 100 - data.rebound_rate[4] ], 
                            name= data.rebound_rate.index[4]),
                     2, 2)
fig3.add_trace(go.Pie(labels = labels, values=[data.rebound_rate[5], 100 - data.rebound_rate[5] ], 
                            name =data.rebound_rate.index[5]),
                     2, 3)
fig3.add_trace(go.Pie(labels = labels, values=[data.rebound_rate[6], 100 - data.rebound_rate[6] ], 
                            name= data.rebound_rate.index[6]),
                     3, 2)

# updating the plots by creating a hole in the middle and specifying the info displayed when hovering.
fig3.update_traces(hole=.45, hoverinfo="label+value+name")

# updating the layout with many parameters and a title 
fig3.update_layout( title_text="REBOUND RATE ACROSS COUNTRIES", title_font={'size':24}, title_x=0.27, 
                   title_y=0.95 , autosize=True,
    
    # annotations in the center of the donut pies.
    annotations=[dict(text= data.rebound_rate.index[0], x=0.11, y=0.88, font_size=13, showarrow=False),
                 dict(text= data.rebound_rate.index[1], x=0.5, y=0.88, font_size=13, showarrow=False),
                 dict(text= data.rebound_rate.index[2], x=0.88, y=0.88, font_size=13, showarrow=False),
                 dict(text= data.rebound_rate.index[3], x=0.1, y=0.5, font_size=13, showarrow=False),
                 dict(text= data.rebound_rate.index[4], x=0.5, y=0.5, font_size=13, showarrow=False),
                 dict(text= data.rebound_rate.index[5], x=0.9, y=0.5, font_size=13, showarrow=False),
                 dict(text= data.rebound_rate.index[6], x=0.5, y=0.12, font_size=13, showarrow=False)])


# creating a dataframe composed of the two last columns
df = data[['Pages_per_session','average_session_duration']]
df

#figure 4

# Create figure with secondary y-axis
fig4 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces (each for one of the columns separately, but within the same figure)
fig4.add_trace(
    go.Scatter(x=df.index, y=df.Pages_per_session, name="Pages per Session (in average)"),
    secondary_y=False )

fig4.add_trace(
    go.Scatter(x=df.index, y=df.average_session_duration, name="Average Session Duration"),
    secondary_y=True )

# Add figure title and specify size and positioning
fig4.update_layout(
    title_text="SESSION DURATION & PAGES PER SESSION", 
    title_font={'size':24}, title_x=0.5, autosize=False, width=900, height=500 )

# Set x-axis title
fig4.update_xaxes(title_text=" <b> Counties </b>")

# Set the two y-axes titles
fig4.update_yaxes(title_text="<b> Pages per Session (in average) </b> ", secondary_y=False)
fig4.update_yaxes(title_text="<b> Average Session Duration </b>", secondary_y=True)


"""### --- 2nd Dataset : Customers Dataset

"""

# calling the dataset previously imported
data2

"""### First, Let's clean the data, we need to:
- extract the date only out of the "last_visit" column
- convert the "subscribed" column into boolean
"""

# extract the date only out of the "last_visit" column
for i in range (len(data2)): 
  data2['last_visit'][i] = data2['last_visit'][i].split('T')[0]

#convert the "subscribed" column into boolean
data2['subscribed'] = data2['subscribed'].isnull().values
data2['subscribed']

# extracting the frequency of Trues and Falses
freq_subs = data2['subscribed'].value_counts()
freq_subs

# Figure 5

# Subscription ratio across platform visitors
subs_ratio = px.bar(freq_subs)

# Add figure title and specify size and positioning
subs_ratio.update_layout(
    title_text="SUBSCRIPTION RATIO ACROSS PLATFORM USERS", 
    title_font={'size':20}, title_x=0.5, autosize=False, width=600, height=400 , plot_bgcolor='white')

# Set x-axis title
subs_ratio.update_xaxes(title_text=" <b> Subscription Status </b>")

# Set the y-axes titles
subs_ratio.update_yaxes(title_text="<b> Number of Visitors </b> ")


### Now, we create a pivot table with unique last_visit dates in the Index and the number of orders in the values 

visits_trend = data2.pivot_table(index=['last_visit'], aggfunc=np.sum )
visits_trend  = pd.DataFrame(visits_trend) 
visits_trend

# Eliminating the last 3 rows (outliers)
visits_trend = visits_trend.drop(visits_trend.index[-3:])

# Figure 6

# Orders and Subscriptions 

visits_trend_fig1 = go.Figure()


visits_trend_fig1.add_trace(go.Scatter(x=visits_trend.index , y=visits_trend.nbr_orders, 
                        mode='lines', name='Total number of Orders', 
                        line=dict(color='green', width=2)))

visits_trend_fig1.add_trace(go.Scatter(x=visits_trend.index , y=visits_trend.subscribed, 
                        mode='lines', name='Number of Subscriptions',
                        line=dict(color='firebrick', width=2)))



visits_trend_fig1.update_layout( title='STATS RELATED WITH DAILY VISITS', xaxis_title='<b> Date </b>', yaxis_title='<b> Frequency </b>' , 
                         title_font={'size':22}, title_x=0.5 , title_y = 0.85,
                         xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', 
                                    linewidth=2, ticks='outside', tickfont=dict( family='Arial', size=12, 
                                                                                color='rgb(82, 82, 82)')),
                         autosize=True, margin=dict(autoexpand=True), showlegend=True, plot_bgcolor='white')


## Again, we create a pivot table with unique country_codes an Index while aggregating :
# the number of subscriptions
# the number of orders
# the amounts paid 

contribution_by_country = data2.pivot_table(index=['country_code'], aggfunc=np.sum )
contribution_by_country  = pd.DataFrame(contribution_by_country) 
contribution_by_country

# Figure 7 

# creating a figure object and setting up the title
visits_trend_fig2 = go.Figure(layout=dict(title=dict(text="SERVICE PERFORMANCE ACROSS COUNTRIES")))

visits_trend_fig2.add_trace(go.Bar(
    x= contribution_by_country.index,
    y= contribution_by_country.nbr_orders,
    name='Number of Orders',
    marker_color='orange'
))

visits_trend_fig2.add_trace(go.Bar(
    x= contribution_by_country.index,
    y= contribution_by_country.subscribed,
    name='number of subscriptions',
    marker_color='blue' ))

# Here we update the layout through fixing different params, we also modify the tickangle of the xaxis, resulting in rotated labels.
visits_trend_fig2.update_layout(barmode='group', xaxis_tickangle=45, plot_bgcolor='white', 
                  autosize=False, width=1200, height=500 , 
                  title_font={'size':22}, title_x=0.5 )


# Figure 8 

# Amount paid by customers on a daily basis

amount_paid_perday = go.Figure()


amount_paid_perday.add_trace(go.Scatter(x=visits_trend.index , y=visits_trend.amount_paid, 
                        mode='lines+markers', name='Daily total paid amount', 
                        line=dict(color='blue', width=2)))


amount_paid_perday.update_layout( title='AMOUNT PAID BY OVERALL CUSTOMERS DAILY', title_font={'size':22}, title_x=0.5 , title_y = 0.85,
                                       xaxis_title='<b> Date </b>', yaxis_title='<b> Amount (in €) </b>', 
                                       xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', 
                                                  linewidth=2, ticks='outside', tickfont=dict( family='Arial', size=12, 
                                                                                              color='rgb(82, 82, 82)')),
                                       autosize=True, margin=dict(autoexpand=True), showlegend=True, plot_bgcolor='white')



# Sorting the amounts paid within each country in an assending manner, and sellecting only the top 15 country to vizualize them
top_revenue_sources= contribution_by_country.amount_paid.sort_values(ascending= False).head(15)
top_revenue_sources

# Figure 9 

# TOP 15 COUNTRIES WITH HIGHEST REVENUES CONTRIBUTION (in €) 
# top_revenue_sources.iloc[index] returns the value corresponding to that index (exp: 24723.3 for the case of Index = 0 )
Revenues_distribution_per_country = go.Figure(data=[go.Pie(labels=top_revenue_sources.index,
                                    values=[top_revenue_sources.iloc[0], top_revenue_sources.iloc[1],
                                            top_revenue_sources.iloc[2],top_revenue_sources.iloc[3],
                                            top_revenue_sources.iloc[4],top_revenue_sources.iloc[5],
                                            top_revenue_sources.iloc[6],top_revenue_sources.iloc[7],
                                            top_revenue_sources.iloc[8],top_revenue_sources.iloc[9],
                                            top_revenue_sources.iloc[10],top_revenue_sources.iloc[11],
                                            top_revenue_sources.iloc[12],top_revenue_sources.iloc[13],
                                            top_revenue_sources.iloc[14]                                 ]  , hole=.3 , 
                                                                                                                   )],
                                              layout=dict(title=dict(text="TOP 15 COUNTRIES WITH HIGHEST REVENUES CONTRIBUTION (in €) ")))

#hover info, text size, pull amount for each pie slice, and stroke
Revenues_distribution_per_country.update_traces(hoverinfo='label+value', textposition='inside',
                  textinfo=' label+percent',
                  marker=dict(line=dict(color='#FFFFFF', width=2)))

Revenues_distribution_per_country.update_layout(title_font={'size':22}, title_x=0.5, autosize=False, width=900, height=500 ) 

Revenues_distribution_per_country

# setting a filter to find the number of users with 0 orders  
null_orders = data2[ data2.nbr_orders == 0 ]
len(null_orders)

# Figure 10 

passive_users_ratio = go.Figure(data=[go.Pie(labels=['Customers with at least one successful order','Customers with no single order'],
                                             values=[ len(data2) - len(null_orders) , len(null_orders) ]  , hole=.3 )],
                                layout=dict(title=dict(text="PASSIVE CUSTOMERS RATIO")))

#hover info, text size, pull amount for each pie slice, and stroke
passive_users_ratio.update_traces(hoverinfo='label+value',
                  textinfo='percent',
                  marker=dict(line=dict(color='white', width=2)))

passive_users_ratio.update_layout(title_font={'size':22}, title_x=0.5, autosize= False, width=700, height=450, ) 


# creating a pivot table with the sum of amounts paid by each user
amount_paid_per_top_users = data2.pivot_table(index=['Name'], values=['amount_paid'], aggfunc=np.sum )

# Saving the pivot table within a dataframe for the sake of easy manipulation 
amount_paid_per_top_users = pd.DataFrame(amount_paid_per_top_users) 

# Sorting the values in order to rank them from least to top spending customer
# selecting the top 10 paying customers (whom are in the bottom of the list in this case since we sorted them ascendingly)
amount_paid_per_top_users = amount_paid_per_top_users.sort_values(by='amount_paid', ascending= True ).tail(10)

# rounding the values
amount_paid_per_top_users['amount_paid'] = round(amount_paid_per_top_users['amount_paid'])
amount_paid_per_top_users

# Figure 11
# Top paying customers

# creating a figure object and setting up the title and the bar plot orientation (h = horisontal bar plot)
top_paying_customers = px.bar(amount_paid_per_top_users,  title='TOP 10 SPENDING USERS LEADERBOARD', orientation='h')

# updating the layout with different parameters
top_paying_customers.update_layout(title_font={'size':22}, title_x=0.5 , plot_bgcolor='white',showlegend=False,
                   xaxis_title_text=' <b> Total amount paid in € </b>', yaxis_title_text=' <b> TOP 10 SPENDERS </b> ', 
                   autosize=False, width=900, height=500 )
# updating the trace colour
top_paying_customers.update_traces(marker_color=  px.colors.sequential.Plasma)

top_paying_customers

"""### --- 3rd Dataset : Sales Across Time (time series)"""

#renaming columns for easier manipulation
data3.rename(columns= {'Date':'date', 'Net Sales':'net_sales'},  inplace = True)

# setting the date as an index
data3.set_index('date', inplace = True)
data3.head()

# Filtering the Series from values coming after the 07-07-2021 , which are all null. 
data3 = data3[ data3.net_sales != 0 ]

# maaking a copy of the clean data at this stage in case it was needed
data3_copy = data3

# Figure 12

# Orders and net sales 

sales_trend = go.Figure()


sales_trend.add_trace(go.Scatter(x=data3.index , y=data3.net_sales, 
                        mode='lines', name='Daily Net Sales',
                        line=dict(color='firebrick', width=3)))


sales_trend.update_layout( title='<b> DAILY NET SALES TREND </b> ', xaxis_title='<b> Date </b>', yaxis_title='<b> Amount in €  </b>' , 
                         title_font={'size':22}, title_x=0.5 , title_y = 0.85, xaxis_tickangle=-45, plot_bgcolor ='white',
                         xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', 
                                    linewidth=4, ticks='inside', tickfont=dict( family='Arial', size=12, 
                                                                                color='rgb(82, 82, 82)')),
                         autosize=True, margin=dict(autoexpand=True), showlegend=True)



"""- As shown by the lineplot, the Net Sales Trend in continuously decreasing and going null after the 7/7/2021 , hence, foreceasting future trend in this case won't be of any added value as the sales trend is not showing fluctuations and the data feeding the model are highly correlated.

### --- 5th Dataset : Average daily session length per user across time (time series)
"""

data5.rename(columns= {'Index des jours':'date'},  inplace = True)
data5.set_index('date', inplace = True)
data5.head()

#Coverting the session length from format hh:mm:ss to seconds
import datetime

for i in range (len(data5)-1):
  h,m,s = data5.Average_timespent_per_person[i].split(':')
  data5.Average_timespent_per_person[i] = int(datetime.timedelta(hours=int(h),minutes=int(m),seconds=int(s)).total_seconds())

data5.head()

# Figure 13 

# Daily  Average Session Duration per User: 


session_duration = go.Figure()


session_duration.add_trace(go.Scatter(x=data5.index , y=data5.Average_timespent_per_person, mode='lines', name='Session Duration',
                                      line=dict(color='indigo', width=2)))


session_duration.update_layout( title='<b> AVERAGE DAILY TIME SPENT PER SINGLE USER </b> ', 
                               xaxis_title='<b> Date </b> ', yaxis_title='<b> Time spent (in seconds)  </b>' , 
                               title_font={'size':22}, title_x=0.5 , title_y = 0.85, xaxis_tickangle=-45, plot_bgcolor ='white',
                               xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', 
                                          linewidth=4, ticks='outside', tickfont=dict( family='Arial', size=12, color='rgb(82, 82, 82)')),
                               autosize=True, margin=dict(autoexpand=True), showlegend=True)



"""### --- 4th Dataset : Daily users flow on the platform (time series)"""

data4.set_index('date', inplace = True)
data4 = data4.dropna()

# making a copy of the cleaned data
data4_copy = data4
data4.head()

# Users flow trend 

users_trend = go.Figure()


users_trend.add_trace(go.Scatter(x=data4.index , y=data4.nbr_users, 
                        mode='lines', name='Nbr of Users',
                        line=dict(color='green', width=2)))


users_trend.update_layout( title='<b> PLATFORM USERS TREND </b> ', xaxis_title=' <b> Date </b>', yaxis_title='<b> Nbr of Users  </b>' , 
                         title_font={'size':22}, title_x=0.5 , title_y = 0.85, xaxis_tickangle=-45, plot_bgcolor ='white',
                         xaxis=dict(showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', 
                                    linewidth=4, ticks='inside', tickfont=dict( family='Arial', size=12, 
                                                                                color='rgb(82, 82, 82)')),
                         autosize=True, margin=dict(autoexpand=True), showlegend=True)




"""# IV - Forecasting Time Series using Deep Learning

## ---------------------------------------------------------------------------------------------------------------
Algorithms: 
- Long-Short Term Memory ( LSTM )
- Gated Recurrent Unit ( GRU )

#### Cases of Daily Paid Amount & Number of Daily Platform Users

## ---------------------------------------------------------------------------------------------------------------

## --- Imports:
"""

#importing required libraries for model training 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler

#for normalizing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

"""## --- Recalling the Two Datasets in Question"""

data4.head()

amount_paid_trend = pd.DataFrame(visits_trend['amount_paid'])
amount_paid_trend_copy = amount_paid_trend
amount_paid_trend.head()

print( 'length of users trend dataset: ' , len(data4)) 
print( 'length of paid amount trend dataset: ' , len(amount_paid_trend))

"""## --- 1 MONTH DAILY USERS FORECAST USING "GRU" """

#extracting values from our dataframe
data4 = data4.values

#creating train and test sets
train = data4[0:362,:]
valid = data4[362:,:]


# Scaling the data: Transforming it into values ranging between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data4)

#converting data4 into x_train and y_train

# Creating a list of incrementing x_train & y_train values across time (each Y is predicted using the 30 past X)
x_train, y_train = [], []
for i in range(30,len(train)):
    x_train.append(scaled_data[i-30:i,0])
    y_train.append(scaled_data[i,0])

#converting X & Y into arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#adding an extra dimention to X train to make it readable by the model
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))

# Creating a list of incrementing x_test & y_test values across time (each Y is predicted using the 30 past X)
x_test, y_test = [], []
for i in range(len(train),len(scaled_data)):
    x_test.append(scaled_data[i-30:i,0])
    y_test.append(scaled_data[i,0])

#converting X & Y into arrays
x_test, y_test = np.array(x_test), np.array(y_test)

#adding an extra dimention to X test to make it readable by the model
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))


#------------------------------------------------------------
# Create GRU model
#model_gru = Sequential()

# Input layer
#model_gru.add(GRU (units = 10, return_sequences = True, input_shape = [x_train.shape[1], x_train.shape[2]]))
#model_gru.add(Dropout(0.2)) 

# Hidden layer
#model_gru.add(GRU(units = 5)) 
#model_gru.add(Dropout(0.2))
#model_gru.add(Dense(units = 1)) 



#Compile model_gru
#model_gru.compile(optimizer='adam',loss='mse')
#history = model_gru.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=1, verbose=0)

#model summary
#model_gru.summary()


#--------------------------------------------------------
# serialize model to JSON
#model_json = model_gru.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model_gru.save_weights("model.h5")
#print("Saved model to disk")
#--------------------------------------------------------

from keras.models import model_from_json 

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


#extracting values from the training set, with unconsidering the last 30 values as they intervene in the prediction of the first y_test values
inputs = data4[len(data4) - len(valid) - 30:]

#Flattening the array (converting a multidimensional array into a 1D array ) 
inputs = inputs.reshape(-1,1)

#Scaling the data
inputs  = scaler.transform(inputs)

#extracting values from the training set, with unconsidering the last 30 values as they intervene in the prediction of the first y_test values
inputs = data4[len(data4) - len(valid) - 30:]

#Flattening the array (converting a multidimensional array into a 1D array ) 
inputs = inputs.reshape(-1,1)

#Scaling the data
inputs  = scaler.transform(inputs)

# creating a loop for dynamic x_test selection, that is updated with every y to be predicted across time
X_test = []
for i in range(30,inputs.shape[0]):
    X_test.append(inputs[i-30:i,0])

# Transforming x_test into an array and extending dimemntions for compatibility purposes with the input format
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

# predicting Y values and descale them for readability purpose (gru_model was changed to loaded_model)
daily_amount = loaded_model.predict(X_test, batch_size=1)



# Root mean squared error - to evaluate the prediction
rms=np.sqrt(np.mean(np.power((valid - daily_amount),2)))
print ('Root mean squared error Value for this model is: ', rms)

# Setting corresponding datavalues to the train batch and test batch (data length is 392, validation set is the last month, so train= 362, test =30 )

train = data4[:362]
valid = data4[362:]

#inversing the transformation executed over the predicted data to get the real values
predictions = scaler.inverse_transform(daily_amount)
daily_amount_gru = daily_amount

# create and fill a train dataframe to be able to vizualize train data with dates
train_df = pd.DataFrame(columns=['date','train_values'])

# adding the train data, previously processed and turned into an np.ndarray type into the train dataframe recenty created, to solve compatibility issues
for i in range (len(train)): 
  train_df = train_df.append({'date': data4_copy.index[i] , 'train_values': int(train[[i]]) }, ignore_index=True)

train_df.tail()

# create and fill a test dataframe to be able to vizualize train data with dates
test_df = pd.DataFrame(columns=['date','valid_values', 'predictions'])


# adding the validation and predicted values witin a test dataframe created for the matter, to solve compatibility issues while visualizing results
for i in range (len(valid)): 
  test_df = test_df.append({'date': data4_copy.index[362 + i] ,
                              'valid_values': int(valid[[i]]) , 
                              'predictions': int(predictions[[i]]) },
                             ignore_index=True)

test_df.head()

final_df = pd.concat([train_df, test_df])
final_df.tail()

# Figure 14

#Plotting the trend of the training set, test set, and predicted values in one plot - GRU
plt.figure(figsize=(14,7))


# DAily users flow forecasting 

users_trend_pred = go.Figure()

users_trend_pred.add_trace(go.Scatter(x = final_df.date, y= final_df.train_values ,
                                            mode='lines', name='Historical users trend'))

users_trend_pred.add_trace(go.Scatter(x = final_df.date, y=final_df.valid_values ,
                                            mode='lines', name='Actual users trend'))

users_trend_pred.add_trace(go.Scatter(x = final_df.date, y=final_df.predictions ,
                                            mode='lines', name='Predicted datapoints'))

users_trend_pred.update_layout( title= f'1-MONTH FORECAST OF TOTAL DAILY USERS TRAFFIC - GRU MODEL (RMSE = {round(rms,2)})', 
                                xaxis_title='<b> Date </b>', yaxis_title='<b>Frequency </b> ' , 
                                title_font={'size':22}, title_x=0.5 , title_y = 0.9, plot_bgcolor ='white',
                                xaxis=dict(showline=True, showgrid=False, showticklabels=True, 
                                           linecolor='rgb(204, 204, 204)', 
                                           linewidth=2, ticks='outside', tickfont=dict( family='Arial', size=12, 
                                                                                color='rgb(82, 82, 82)')),
                                autosize=True, margin=dict(autoexpand=True), showlegend=True )


