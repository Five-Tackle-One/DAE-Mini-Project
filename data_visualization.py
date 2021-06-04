import matplotlib.pyplot as plt
import pandas as pd

def group_title_value(yearly_group,column_name,unique_column,Years = ["2015","2016","2017","2018"],target='Net Income'): # Yearly Group Is A List of Data Frames
    dic = []
    for i in range(len(unique_column)):
        col_year_value = [group_.groupby([column_name]).mean()['Net Income'].values[i] for group_ in yearly_group]
        dic.append(dict(zip(Years,col_year_value)))
    df = pd.DataFrame(dic).T
    for col_value,col_title in zip(df.columns.values,unique_column):
        df.rename(columns={col_value:col_title},inplace=True)
    return df

def plot_dataframes(dataframes,titles,categories,columns,xlabels=["Year","Year","Year"],ylabels=["Mean Net Income","Mean Net Income","Mean Net Income"]):
    fig1, f1_axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True,figsize=(30,10))
    cm = plt.get_cmap('gist_rainbow')
    for plot in range(len(dataframes)):
        dataframes[plot].plot(ax=f1_axes[plot],linewidth=3.0)
        f1_axes[plot].set_title(titles[plot],size=20)
        f1_axes[plot].set_xlabel(xlabels[plot],fontsize=22)
        f1_axes[plot].set_ylabel(ylabels[plot],fontsize=22)
        f1_axes[plot].legend(categories[plot],loc='upper left',prop={'size':17})
    fig1.tight_layout(pad=3.0)
    plt.show()
    
    
def plot_scatter_matrix(data_matrix,Title,x="Head Age",y="Net Income",Colors=["DarkBlue","DarkGreen","Red","Purple"]):
    fig2, f2_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True,figsize=(30,10))
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix[i])):
            ghs_index = data_matrix[i,j][[x,y]] #9999999.0
            threshold_data = ghs_index[ghs_index[y] <= 900000.0 ]
            plot_data = threshold_data[threshold_data[y] > 0]
            plot_data.plot.scatter(x=x,y=y,ax=f2_axes[i][j],color=Colors[i*2 + j])
            f2_axes[i][j].set_title(Title.format(i*2 + j + 5),fontsize=20)
            f2_axes[i][j].set_xlabel("Year",fontsize=15)
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