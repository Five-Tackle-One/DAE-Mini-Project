import matplotlib.pyplot as plt
import pandas as pd


"""
Input: 
yearly_group: List of datasets per year i.e [ 2015 dataset, 2016 dataset ,2017 dataset, 2018 dataset]
column_name: Column Name 
--------------
Return:
a dataset containing the means of the columns given
"""
def group_title_value(yearly_group,column_name,unique_column,Years = ["2015","2016","2017","2018"],target='Net Income'): # Yearly Group Is A List of Data Frames
    dic = []
    for i in range(len(unique_column)):
        col_year_value = [group_.groupby([column_name]).mean()['Net Income'].values[i] for group_ in yearly_group]
        dic.append(dict(zip(Years,col_year_value)))
    df = pd.DataFrame(dic).T
    for col_value,col_title in zip(df.columns.values,unique_column):
        df.rename(columns={col_value:col_title},inplace=True)
    return df



"""
Input:
yearly_group: List of datasets per year i.e [ 2015 dataset, 2016 dataset ,2017 dataset, 2018 dataset]
column_name: Column Name 
--------------
Return:
a dataset containing the means of the numeric columns given
"""
def group_mean_dataset(yearly_group,column_name="Head Age",Years = ["2015","2016","2017","2018"]):
    ghs_means = []
    for yearly_value in yearly_group:
        ghs_means.append(yearly_value[column_name].mean())
    mean_dict = dict(zip(Years,ghs_means))
    mean_df = pd.DataFrame.from_dict(mean_dict, orient='index')
    mean_df.rename(columns={0:"Mean Age"},inplace=True)
    return mean_df
"""
Plot the distribution of the dataframes given in the list

"""
def plot_dataframes(dataframes,titles,categories,columns,xlabels=["Year","Year","Year"],ylabels=["Mean Net Income","Mean Net Income","Mean Net Income"],ncols=3):
    fig1, f1_axes = plt.subplots(ncols=ncols,nrows=1, constrained_layout=True,figsize=(30,10))
    cm = plt.get_cmap('gist_rainbow')
    for plot in range(len(dataframes)):
        dataframes[plot].plot(ax=f1_axes[plot],linewidth=3.0)
        f1_axes[plot].set_title(titles[plot],size=20)
        f1_axes[plot].set_xlabel(xlabels[plot],fontsize=22)
        f1_axes[plot].set_ylabel(ylabels[plot],fontsize=22)
        f1_axes[plot].legend(categories[plot],loc='upper left',prop={'size':17})
    fig1.tight_layout(pad=3.0)
    plt.show()
    


    
"""
The values are already encoded: But we want to visualize the data using labels
This method will take the labels mentioned in the variable description and map them to the encoded values of the data

Input:
Regression Survey Dataset
---------------

Return: A Column of Replaced Values
"""

def prepare_education_gender_plot(regression_survey,col_name="education_level",grade_label = "Grade{}",ntc_label = "NTC{}",degrees = ["Certificate < G12","Diploma < G12","Certificate w/ G12","Diploma w/ G12","Higher Diploma","Post Higher Diploma","Bachelor","Bachelor w/ PG Diploma","Honours","Higher Degree","Other","No Schooling"]):
    grades = [grade_label.format(str(i)) for i in range(13)]
    grades.append("Grade 12 Exemption")
    ntcs = [ntc_label.format(str(i+1)) for i in range(6)]
    reg = regression_survey['education_level'].copy()
    labels = [grades,ntcs,degrees]
    labels = [label for sub in labels for label in sub]
    for label,ri in zip(labels,sorted(reg.unique())):
        reg.replace({ri:label},inplace=True)
    return reg



"""
Get the mean head age of each year and scatter plot against respective year's net income

Input: Matrix of Datasets
------------------------

"""
def plot_scatter_matrix(data_matrix,Title,x="Head Age",y="Net Income",Colors=["DarkBlue","DarkGreen","Red","Purple"]):
    fig2, f2_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True,figsize=(30,10))
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            ghs_index = data_matrix[i,j][[x,y]] #9999999.0
            threshold_data = ghs_index[ghs_index[y] <= 900000.0 ]
            plot_data = threshold_data[threshold_data[y] > 0]
            plot_data.plot.scatter(x=x,y=y,ax=f2_axes[i][j],color=Colors[i*2 + j])
            f2_axes[i][j].set_title(Title.format(i*2 + j + 5),fontsize=20)
            f2_axes[i][j].set_xlabel("Age",fontsize=15)
            f2_axes[i][j].set_ylabel(y,fontsize=17)
            

def display_records_per_year(records,key_string = "201{} House Survey"):
    key_string = "201{} House Survey"
    ghs_dict = {}
    ghs_keys = [key_string.format(i) for i in range(5,9)]
    ghs_records = [len(ghs_i) for ghs_i in records]
    ghs_dict = dict(zip(ghs_keys,ghs_records))
    ghs_pd = pd.DataFrame.from_dict(ghs_dict,orient='index')
    ghs_pd.rename(columns={0:"Number of Records"},inplace=True)
    return ghs_pd