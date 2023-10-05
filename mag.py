#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Xlayoubi18(sand, orm, Co, Cu, Fe, Zn, pH, np):
    """
    Ayoubi 2018a pedotranfer equation for contaminated soils Table 3
    """
    return np.exp( -24.41 + 0.976*np.log(Fe) + 1.091*np.log(sand) + 0.467*np.log(Cu) + 2.211*np.log(Co) + 3.226*np.log(pH) - 0.441*np.log(orm) + 0.333*np.log(Zn))


def Xlayoubi18pli(PLI):
    """
    Ayoubi 2018a pedotranfer equation for contaminated soils Figure 4
    """
    return (PLI -1.5801)/0.0006


def phosphate_x(Pads):
    """
    Camargo et al 2016.  A total of 308 soil samples were collected from Hapludox and Eutrudox soils formed from sandstone in Brazil.
    """
    return (Pads - 379.182)/17.213


def phosphate_fe(Pads):
    """
    Camargo et al 2016.  A total of 308 soil samples were collected from Hapludox and Eutrudox soils formed from sandstone in Brazil.
    Fe is iron extracted by dithionite–citrate–bicarbonate (Fed),
    """
    return ((Pads - 214.36)/54.086)*1000


def canbaypli(PLI):
    """
    Canbay et al. 2010 Fig. 6
    The study on topsoil contamination due to heavy metals was carried out by using the Magnetic susceptibility (MS) measurements in Izmit     industrial city, northern Turkey.
    """
    return (PLI - 1.36)/0.01


def canbaypoli(x, Cr, Cu, Pb, Ni, pd):
    """
    Own polinomial propose based on Canbay et al. 2020 dataset
    """
    from sklearn import linear_model

    variables = pd.DataFrame(x, columns=["Cr", "Cu", "Pb", "Ni"])
    variables["Cr2"] = x.Cr**2
    variables["Cu2"] = x.Cu**2
    variables["Pb2"] = x.Pb**2
    variables["Ni2"] = x.Ni**2

    regression = linear_model.LinearRegression()
    model = regression.fit(variables, x.Xlf)
    score = model.score(variables, x.Xlf)
    canbaycoef = model.coef_
    
    xlf = model.intercept_ + canbaycoef[0]*Cr + canbaycoef[1]*Cu + canbaycoef[2]*Pb + canbaycoef[3]*Ni + canbaycoef[4]*(Cr**2) + canbaycoef[5]*(Cu**2) + canbaycoef[6]*(Pb**2) + canbaycoef[7]*(Ni**2)
    return xlf


######################################### # # # # GAMMA RAY SPECTOMETRY      # # # # ###############################################


def petersensand(sand):
     return (sand -81.54) /6.63
    
def petersensilt(silt):
    """
    Petersen et al. 2012. Spectometry in european soils. ThK is the ratio Th / K torium divided by potasium.
    """
    return (silt -25.12) / 2.03

def petersenclay(clay):
    return (clay - 6.67) /4.6

def petersencec(cec):
    return (cec - 3.28) / 3.3

def petersenph(pH):
    return (pH - 7.59) / 0.15

def petersenoc(oc):
    return (oc-0.71) /0.15


###########################################################       GRAPHS      #########################################################


def pltmsc(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, lw, yc, plt):
    ax1.legend(loc='upper right', fontsize = 10)
    ax1.set_title("Susceptibility vs Sand" , fontweight='bold', fontsize=25) 
    ax1.tick_params(axis='y', labelsize=12) 
    ax1.tick_params(axis='x', labelsize=12) 
    ax1.set_xlabel('Sand [%]', fontsize = 16) 
    ax1.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_ylim(0, yc)  
    #ax1.set_xlim(1, 50) 
    ax1.legend(loc='upper right', fontsize = 10)

    ax2.legend(loc='upper left', fontsize = 10) 
    ax2.set_title("Susceptibility vs Organic matter" , fontweight='bold', fontsize=25) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    ax2.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax2.set_xlabel('Organic matter [%]', fontsize = 16) 
    ax2.grid(True) 
    ax2.set_ylim(0, yc) 
    #ax2.set_xlim(10, 100000)
    ax2.legend(loc='upper left', fontsize = 10) 

    ax3.grid(True) 
    ax3.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax3.set_xlabel("Cobalt [mg/Kg]" , fontsize = 16) 
    ax3.legend(loc='upper left', fontsize = 10) 
    ax3.set_title("Susceptibility vs Cobalt" , fontweight='bold', fontsize=25) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    ax3.set_ylim(0, yc) 
    #ax3.set_xlim(0, 60) 
    ax3.legend(loc='upper left', fontsize = 10) 

    ax12.legend(loc='upper right', fontsize = 10) 
    ax12.set_title("Susceptibility vs Copper" , fontweight='bold', fontsize=25) 
    ax12.tick_params(axis='y', labelsize=12) 
    ax12.tick_params(axis='x', labelsize=12) 
    ax12.set_xlabel('Copper [mg/Kg]', fontsize = 16) 
    ax12.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax12.grid(True) 
    ax12.set_ylim(0, yc) 
    #ax12.set_xlim(0.1, 0.8) 
    ax12.legend(loc='upper right', fontsize = 10) 

    ax5.legend(loc='upper right', fontsize = 10) 
    ax5.set_title("Susceptibility vs Iron" , fontweight='bold', fontsize=25) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_xlabel('Iron [mg/Kg]', fontsize = 16) 
    ax5.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_ylim(0, yc) 
    #ax5.set_xlim(0, 25) 
    ax5.legend(loc='upper right', fontsize = 10) 

    ax6.set_title("Susceptibility vs Zinc" , fontweight='bold', fontsize=25) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax6.set_xlabel('Zinc [mg/Kg]', fontsize = 16) 
    ax6.grid(True) 
    #ax6.set_xlim(0, 65) 
    ax6.set_ylim(0, yc)
    ax6.legend(loc='upper right', fontsize = 10) 
    
    ax7.set_title("Susceptibility vs pH" , fontweight='bold', fontsize=25) 
    ax7.tick_params(axis='y', labelsize=12) 
    ax7.tick_params(axis='x', labelsize=12) 
    ax7.set_xlabel('pH', fontsize = 16) 
    ax7.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax7.grid(True) 
    ax7.set_ylim(0, yc) 
    #ax7.set_xlim(0, 25) 
    ax7.legend(loc='upper right', fontsize = 10) 

    ax8.set_title("LF Susceptibility vs PLI" , fontweight='bold', fontsize=25) 
    ax8.tick_params(axis='y', labelsize=12) 
    ax8.tick_params(axis='x', labelsize=12) 
    ax8.set_ylabel('LF Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax8.set_xlabel("Pollution Load Index", fontsize = 16) 
    ax8.grid(True) 
    #ax8.set_xlim(0, 65) 
    #ax8.set_ylim(0, yc)
    ax8.legend(loc='upper right', fontsize = 10) 
    
    ax9.legend(loc='upper right', fontsize = 10) 
    ax9.set_title("Susceptibility vs Chrome" , fontweight='bold', fontsize=25) 
    ax9.tick_params(axis='y', labelsize=12) 
    ax9.tick_params(axis='x', labelsize=12) 
    ax9.set_xlabel('Chrome [mg/Kg]', fontsize = 16) 
    ax9.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax9.grid(True) 
    ax9.set_ylim(0, yc) 
    #ax9.set_xlim(0, 25) 
    ax9.legend(loc='upper right', fontsize = 10) 

    ax10.set_title("Susceptibility vs Plumb" , fontweight='bold', fontsize=25) 
    ax10.tick_params(axis='y', labelsize=12) 
    ax10.tick_params(axis='x', labelsize=12) 
    ax10.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax10.set_xlabel('Plumb [mg/Kg]', fontsize = 16) 
    ax10.grid(True) 
    #ax10.set_xlim(0, 65) 
    ax10.set_ylim(0, yc)
    ax10.legend(loc='upper right', fontsize = 10) 
    
    ax11.set_title("Susceptibility vs Nickel" , fontweight='bold', fontsize=25) 
    ax11.tick_params(axis='y', labelsize=12) 
    ax11.tick_params(axis='x', labelsize=12) 
    ax11.set_xlabel('Nickel [mg/Kg]', fontsize = 16) 
    ax11.set_ylabel('Susceptibility [10−8 m3/kg]', fontsize = 16) 
    ax11.grid(True) 
    ax11.set_ylim(0, yc) 
    #ax11.set_xlim(0, 25) 
    ax11.legend(loc='upper right', fontsize = 10) 

def sets(axc1, axc2, axc3):
    wide = 90
    axc1.legend(loc='lower right', fontsize = 10)
    axc1.set_title("Field 1", fontweight='bold', fontsize=25)
    axc1.tick_params(axis='y', labelsize=12)
    axc1.tick_params(axis='x', labelsize=12)
    axc1.set_xlabel('Magnetic susceptibility [e-5]', fontweight='bold', fontsize = 16)
    axc1.set_ylabel('Depth [cm]', fontweight='bold', fontsize = 16)
    axc1.grid(True)
    #axc1.set_ylim(0, 65) 
    axc1.set_xlim(1, wide)

    axc2.legend(loc='lower right', fontsize = 10)
    axc2.set_title("Field 2", fontweight='bold', fontsize=25)
    axc2.tick_params(axis='y', labelsize=12)
    axc2.tick_params(axis='x', labelsize=12)
    axc2.set_xlabel('Magnetic susceptibility [e-5]', fontweight='bold', fontsize = 16)
    #axc2.set_ylabel('Depth [cm]', fontsize = 16)
    axc2.grid(True)
    #axc2.set_ylim(0, 65)
    axc2.set_xlim(1, wide)

    axc3.grid(True)
    axc3.set_xlabel('Magnetic susceptibility [e-5]', fontweight='bold', fontsize = 16)
    #axc3.set_ylabel('Depth [cm]', fontsize = 16)
    axc3.legend(loc='lower right', fontsize = 10)
    axc3.set_title("Field 3", fontweight='bold', fontsize=25)
    axc3.tick_params(axis='y', labelsize=12)
    axc3.tick_params(axis='x', labelsize=12)
    #axc3.set_ylim(0, 65)
    axc3.set_xlim(1, wide)
    
def camargoplots(axp1, axp2):
    axp1.legend(loc='upper right', fontsize = 10)
    axp1.set_title("Susceptibility vs Pads" , fontweight='bold', fontsize=25) 
    axp1.tick_params(axis='y', labelsize=12) 
    axp1.tick_params(axis='x', labelsize=12) 
    axp1.set_xlabel('Phosphate adsorved [mg/Kg]', fontsize = 16) 
    axp1.set_ylabel('Susceptibility [10−6 m3/kg]', fontsize = 16) 
    axp1.grid(True) 
    #axp1.set_ylim(0, yc)  
    #axp1.set_xlim(1, 50) 
    axp1.legend(loc='upper right', fontsize = 10)

    axp2.legend(loc='upper right', fontsize = 10)
    axp2.set_title("Fe vs Pads" , fontweight='bold', fontsize=25) 
    axp2.tick_params(axis='y', labelsize=12) 
    axp2.tick_params(axis='x', labelsize=12) 
    axp2.set_xlabel('Phosphate adsorved [mg/Kg]', fontsize = 16) 
    axp2.set_ylabel('Fe [mg/Kg]', fontsize = 16) 
    axp2.grid(True) 
    #axp2.set_ylim(0, yc)  
    #axp2.set_xlim(1, 50) 
    axp2.legend(loc='upper right', fontsize = 10)
    
def pltgamma(ax1, ax2, ax3, ax4, ax5, ax6, lw, yc, plt):
    
    ax1.set_title("Sand vs Th/K" , fontweight='bold', fontsize=25) 
    ax1.set_ylabel('Th / K ratio', fontsize = 16) 
    ax1.set_xlabel('Sand [%]', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_ylim(0, yc)  
    #ax1.set_xlim(1, 50) 
    ax1.legend(loc='upper right', fontsize = 10)

    ax2.set_title("Silt vs Th/K" , fontweight='bold', fontsize=25) 
    ax2.set_xlabel('Silt [%]', fontsize = 16) 
    ax2.set_ylabel('Th / K ratio', fontsize = 16) 
    ax2.legend(loc='upper left', fontsize = 10) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    ax2.grid(True) 
    ax2.set_ylim(0, yc) 
    #ax2.set_xlim(10, 100000)
    ax2.legend(loc='upper left', fontsize = 10) 

    ax3.set_xlabel('Clay [%]', fontsize = 16) 
    ax3.set_ylabel("Th / K ratio" , fontsize = 16) 
    ax3.set_title("Clay vs Th/K" , fontweight='bold', fontsize=25) 
    ax3.grid(True) 
    ax3.legend(loc='upper left', fontsize = 10) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    ax3.set_ylim(0, yc) 
    #ax3.set_xlim(0, 60) 
    ax3.legend(loc='upper left', fontsize = 10) 

    ax4.legend(loc='upper right', fontsize = 10) 
    ax4.set_title("CEC vs Th/K" , fontweight='bold', fontsize=25) 
    ax4.tick_params(axis='y', labelsize=12) 
    ax4.tick_params(axis='x', labelsize=12) 
    ax4.set_ylabel('Th / K ratio', fontsize = 16) 
    ax4.set_xlabel('Cation Exchange Capacity [cmol/Kg]', fontsize = 16) 
    ax4.grid(True) 
    ax4.set_ylim(0, yc) 
    #ax4.set_xlim(0.1, 0.8) 
    ax4.legend(loc='upper right', fontsize = 10) 

    ax5.legend(loc='upper right', fontsize = 10) 
    ax5.set_title("pH vs Th/K" , fontweight='bold', fontsize=25) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_ylabel('Th / K ratio', fontsize = 16) 
    ax5.set_xlabel('pH', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_ylim(0, yc) 
    #ax5.set_xlim(0, 25) 
    ax5.legend(loc='upper right', fontsize = 10) 

    ax6.set_title("Organic content vs Th/K" , fontweight='bold', fontsize=25) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_xlabel('Organic content [%]', fontsize = 16) 
    ax6.set_ylabel('Th / K ratio', fontsize = 16) 
    ax6.grid(True) 
    #ax6.set_xlim(0, 65) 
    ax6.set_ylim(0, yc)
    ax6.legend(loc='upper right', fontsize = 10) 