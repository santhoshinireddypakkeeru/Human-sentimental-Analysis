# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:11:55 2021

@author: Santhoshini
"""

import pandas as pd
import pickle
import webbrowser

import dash
import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np
import plotly.express as px


#declaring global variables

app = dash.Dash()
projectName = None

#loading module

def load_model():
    global df
    df = pd.read_csv('etsy_reviews.csv')
    
    global total_reviews
    total_reviews = df.shape[0]
    
    global pickle_model
    file = open("pickle_model.pkl", "rb")
    pickle_model = pickle.load(file)
    
    global vocab
    file = open("feature.pkl", "rb")
    vocab = pickle.load(file)


    global df_balanced_reviews
    df_balanced_reviews = pd.read_csv('balanced_reviews.csv')
    df_balanced_reviews.dropna(inplace = True)
    df_balanced_reviews = df_balanced_reviews[df_balanced_reviews['overall'] != 3]
    df_balanced_reviews['reviewType'] = np.where(df_balanced_reviews['overall'] >3, 'Positive', 'Negative')
        
 #open browser
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')
#checking whether the review is positive or negative

def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error = "replace", vocabulary = vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    
    return pickle_model.predict(reviewText)

#creating application user interface
def create_app_ui():
    
    main_Layout = html.Div(
     [
          html.H1(
              children = "Sentiment Analysis With Insights",
              id = "Main_title",
             
              style ={
                      'textAlign': 'center',
                      'color': '#FOFOFO',
                      'background-color':'#FOFOFO',
                
                      }
          ),
      
        html.Div(
            [
                
                html.P(children = "Balanced Reviews Pie Chart", id = "heading1",
                        style = {
                            'textAlign':'center',
                            'font-size':'30px',
                            'color':'#EEFBFB',
                            'background-color':'#12232E'}),
                dcc.Graph(
                        figure = px.pie(
                        data_frame = df_balanced_reviews,
                        names = 'reviewType',
                        color_discrete_sequence = ["red", "green"],
                        width = 500,
                        height = 400,
                        
                    ),
                    style = {
                        'display': 'flex',
                        'width':'100%',
                        'justify-content':'center',
                        'background-color': '#DE3163'
                        },
                )
        
            ],
            
            style = {
                #'border-style':'outset',
                #'border-color':'#203647',
                }
        ),
        
        html.Div(
            [
                html.P(children='Select a review', id='select_review',
                       style ={
                           'text-align':'center',
                           'font-size':'30px',
                           'color': '#EEFBFB',
                           'background-color':'#12232E'}),   
                dcc.Dropdown(
                    id='dropdown',
                    placeholder = "Select something",
                    options = [
                            {'label': df.iloc[i]['reviewText'], 'value': df.iloc[i]['reviewText']} for i in range(0, total_reviews)],
                    optionHeight = 100,
                    style = {
                        'color': '#007CC7',
                        'background-color': '#EEFBFB'}
                ), 
                
                html.Div(
                    
                    html.P(children = "Review Type: ", style = {
                        'color': '#EEFBFB',
                        'font-size': '20px'}),
                    style = {
                        'width':'49%',
                        'display':'inline-block',
                        }
                ),
                html.Div(
                    
                    html.P(children = None, id = "result_dropdown", style ={
                        'color': '#EEFBFB',
                        'font-size':'20px'}),
                    style = {
                        'display':'inline-block',
                        }
                )
                
        ],
            style = {
                #'border-style':'outset',
                #'border-color':'#4DA8DA'
                }
        ),
        
        html.Div(
            [
                html.P(children='Text analysis', id='text_analysis',
                       style = {
                           'text-align':'center',
                           'font-size':'30px',
                           'background-color': '#12232E',
                           'color':'#EEFBFB'}),
                dcc.Textarea(
                    id = 'textarea_review',
                    placeholder='Enter the review here...',
                    style={'width': '100%', 
                           'height': '50%', 'justify-contents':'center',
                           'display':'flex',
                           'background-color': '#12232E',
                           'color':'#EEFBFB'}
                ),
                
                html.Div(
                  html.Button(
                      children = 'Analyse Sentiment',
                              id = 'button_review',
                              n_clicks = 0,
                              style = {
                                  'background-color': '#203647',
                                  'color': '#EEFBFB'}
                              ),
                  style = {
                      'text-align':'center',
                      'padding': '5px',
                      }
                    ),
                
                html.Div(    
                    html.P(children = "Review Type: "),
                    style = {
                        'width': '49%',
                        'display':'inline-block',
                        'font-size':'20px',
                        'color':'#EEFBFB'
                        }
                ),
                
                html.Div(    
                    html.P(children = None, id = "result_text"),
                    style = {
                        'display':'inline-block',
                        'font-size':'20px',
                        'color':'#EEFBFB'},
                    
                )
                
            ],
            style = {
                    #'border-style':'outset',
                    #'border-color':'#EEFBFB'
                    }
        )
     ],
     style = {
         'background-color':'#DE3163'
         }
     
    )

    
    return main_Layout

@app.callback(   
    Output('result_dropdown', 'children'),
    [Input('dropdown', 'value')]
    )

def update_dropdown_ui(dropdown_value):
    
    result_list1 = check_review(dropdown_value)
        
    if (result_list1[0] == 0):
        result1 = 'Negative'
    elif (result_list1[0] == 1):
        result1 = 'Positive'
    else:
        result1 = 'Unknown'
    
    return result1

'''@app.callback(
                Output('result_text', 'children'),
                [
                    Input('button_review', 'n_clicks'),
                ],
                State('textarea_review', 'value')
            )


def update_app_ui(n_clicks, textarea_value):
    
    if(n_clicks > 0):
        result_list2 = check_review(textarea_value)
        if (result_list2[0] == 0):
            result2 = 'Negative'
        elif (result_list2[0] == 1):
            result2 = 'Positive'
        else:
            result2 = 'Unknown'
            
    return result2'''

#main function

def main():
    load_model()
    open_browser()
    
    global projectName
    projectName = "Sentiment Analysis With Insights"
    
    global app
    app.layout = create_app_ui()
    app.title = projectName
    app.run_server()
    
    projectName = None
    app = None
    
if __name__ == '__main__':
        main()