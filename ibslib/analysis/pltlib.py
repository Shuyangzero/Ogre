

"""
Holds functions for often used formating features for plotting. 

README:
    Definition of common arguments:
        ____________________________________________________________________
        | Argument  |           Example          | Definiton                |
        |-------------------------------------------------------------------|
        |    ax     | ax = fig.add_subplot(111)  | Matplotlib Axes object   |
        |   pltn    | plt1 = plt.scatter(x,y)    | Matplotlib Collection    |
        |  plt_list | plt_list = plt1+plt2       | Concatenated Collection  | 
        |-------------------------------------------------------------------|
"""

# Default formating parameters
d_label_size=14
d_tick_size=12
d_line_width = 3
d_figure_size=(10,8)
d_labelpad_x = 5
d_labelpad_y1 = 5
d_labelpad_y2 = 20
d_tick_width=2
d_capsize=0
d_errorevery=10
d_elinewidth=0
d_grid=False

###############################################################################
# Aesthetics                                                                  # 
###############################################################################

def format_ticks(ax,
                   linewidth=d_line_width,
                   tick_width=d_tick_width,
                   tick_size=d_tick_size,
                   grid=d_grid):
    ax.spines['top'].set_linewidth(tick_width)
    ax.spines['right'].set_linewidth(tick_width)
    ax.spines['bottom'].set_linewidth(tick_width)
    ax.spines['left'].set_linewidth(tick_width)
    ax.tick_params(axis='both', which='major', labelsize=tick_size,
                   width=2, length=7)
    ax.grid(grid, axis='both', which='both')


def add_legend(plot_list, ax):
    ''' 
    Pass in plot_list constructed most easily with:
            plt_1 = plt.scatter
            plt_2 = plt.scatter
            plot_list = plt_1 + plt2
            
    ax = fig.add_subplot(111)
    '''
    labs = [p.get_label() for p in plot_list]
    ax.legend(plot_list, labs, loc=0)

def add_legend_by_handle(ax, handles, fontsize=d_label_size):
    """
    Add all handles to a legend for Axis ax
    """
    pass
    

def format_axis(ax, xlabel='', ylabel='', label_size=d_label_size,
                labelpad_x=d_labelpad_x, labelpad_y=d_labelpad_y1):
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel, fontsize=label_size, labelpad=labelpad_x)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel, fontsize=label_size, labelpad=labelpad_y)
    

###############################################################################
# Data (?)                                                                    # 
###############################################################################
    
class data_class:
    """
    Class to store data that was used for plotting. 
    This will be returned from plotting functions so the data is usable after
    """
