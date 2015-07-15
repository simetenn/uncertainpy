import matplotlib.pyplot as plt

def prettyPlot(x, y, title = None, xlabel = None, ylabel = None, color = 0):
    axis_grey = (0.5,0.5,0.5)
    titlesize = 18
    fontsize = 16
    labelsize = 14
    # These are the "Tableau 20" colors as RGB.  
    tableau20 = [(31, 119, 180), (14, 199, 232), (255, 127, 14), (255, 187, 120),  
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
  
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.)  

    if plt.gcf() == "None":
        plt.figure(figsize=(10, 7.5))
    else:
        plt.clf()
    
    ax = plt.subplot(111)

    #for spine in ax.spines:
    #    ax.spines[spine].set_edgecolor(axis_grey)

    ax.spines["top"].set_edgecolor("None") 
    ax.spines["bottom"].set_edgecolor(axis_grey)
    ax.spines["right"].set_edgecolor("None")
    ax.spines["left"].set_edgecolor(axis_grey)


    ax.tick_params(axis="x", which="both", bottom="on", top="off",  
                    labelbottom="on", color=axis_grey, labelcolor="black",
                    labelsize=labelsize)  
    ax.tick_params(axis="y", which="both", right="off", left="on",  
                    labelleft="on", color=axis_grey, labelcolor="black",
                    labelsize=labelsize)  
        
    ax.plot(x, y, color=tableau20[color], linewidth=2, antialiased=True)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.set_xlim([min(x),max(x)])
    ax.set_ylim([min(y),max(y)])

    
    return ax, tableau20
