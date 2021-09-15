# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:37:47 2021

@author: DELL
"""

import dash
import dash_core_components as dcc
import dash_html_components as html

#ayshek is a python file including all figures used in this web app (originally, was jupyter file)
from ayshek import fig , fig2 ,fig3 ,fig4 , visits_trend_fig1 , visits_trend_fig2, amount_paid_perday, Revenues_distribution_per_country , passive_users_ratio , top_paying_customers , session_duration, users_trend_pred, subs_ratio 
                
#-----------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)


layout = dict( autosize=True, automargin=True, margin=dict( l=30, r=30, b=20, t=40),
              hovermode="closest", plot_bgcolor="#F9F9F9", paper_bgcolor="#F9F9F9" )




app.layout = html.Div([
    
#headers


html.Div([
    
    html.Div([
    html.Div(
        [
            html.H1(
                'E-Business Visualisation & Forecasting Dashboard', 
                style={'text-align':'center', 'color': '#4287f5'} )  ,

            html.H2(
                '              ______________________________________________________________________________________________________________________________________________________________________________' ,
                 style={'text-align':'center'}   )
            ],

        className='eight columns'
        ),
    html.Img(
        src="https://i.ibb.co/PW82CPQ/Amine-Litim.png",
        className='two columns',
        ),
    html.A(
        html.Button(
            "Who Am I?",
            id="who_am_i"

            ),
        href="https://www.linkedin.com/in/litimamine/",
        className="two columns"
        )
    ],
    id="header",
    className='row',
    ),
    
html.H4(
'The following dashboard converts the internal data of a local satrtup into interactive Dash visuals defining diffirent KPIs, summarizing the business Process, and allowing the detection of anomalies to end up supporting the decision-making process.'
'The platform dashboard also includes a sample  NeuralNet-based forecast graph coming under the GRU model (Gated Recurrent Unit) enabeling to predict future daily users traffic, which can be very useful for the business and specefically the marketing department' ,
style={'text-align':'center',}
                ), 


html.H2(
                '              ______________________________________________________________________________________________________________________________________________________________________________' ,
                 style={'text-align':'center'}   ),
    # plots 
    
    html.Div( dcc.Graph( id ='figure1', figure= fig ) ,
             style={'width': '70%', 'display': 'inline-block'}),
    
    html.Div( dcc.Graph( id ='figure2', figure= passive_users_ratio ) ,
             style={'width': '28%', 'display': 'inline-block'}), 
    
    
    ]) , 

html.Div([
    
    
    html.Div( dcc.Graph( id ='figure3', figure= fig4 ) ,
             style={'width': '66%', 'display': 'inline-block'}) ,
    
       html.Div( dcc.Graph( id ='figure4', figure= top_paying_customers ) ,
             style={'width': '33%', 'display': 'inline-block'})  , 
      
    
    ]) , 

html.Div([ 
    
    
   html.Div( dcc.Graph( id ='figure5', figure= fig2 ) ,
             style={'width': '55%', 'display': 'inline-block'}) ,

   html.Div( dcc.Graph( id ='figure6', figure= visits_trend_fig2 ) ,
             style={'width': '40%', 'display': 'inline-block'}) 
   ]) ,


html.Div([ 
    
    html.Div( dcc.Graph( id ='figure7', figure= visits_trend_fig1 ) ,
             style={'width': '78%', 'display': 'inline-block'}),
    
    
    
    html.Div( dcc.Graph( id ='figure8', figure= Revenues_distribution_per_country) ,
             style={'width': '20%', 'display': 'inline-block'})
    
    
    ]) , 
  

html.Div([ 
    
    
   html.Div( dcc.Graph( id ='figure9', figure=  amount_paid_perday) ,
             style={'width': '80%', 'display': 'inline-block'}) ,

   html.Div( dcc.Graph( id ='figure10', figure= subs_ratio) ,
             style={'width': '18%', 'display': 'inline-block'})  
   
   ]) ,


html.Div( 
   html.Div( dcc.Graph( id ='figure11', figure= session_duration) ,
             style={'width': '99%', 'display': 'inline-block'}) ) ,


html.Div( 
    html.Div( dcc.Graph( id ='figure12', figure= users_trend_pred) ,
             style={'width': '98%', 'display': 'inline-block'}) ) ,



html.Div( 
    html.Div( dcc.Graph( id ='figure13', figure= fig3)) ) ,







] ,
    
    style={'align-content': 'center'} )

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)





