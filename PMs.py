"""
Pedotransfer functions
======================

...

"""

# Import
import numpy as np
e =   2.718280828
e_0 = 8.8541878028e-12 # Vacum permittivity
pi =  3.141592653589793


    ################################ ELECTRICAL CONDUCTIVITY EQUATIONS ################################

    
def parallel(vol_moist_cont, bulkdens, partdens, cond_matrix, rest_wat, cond_air = 0.01e-3):
    """
    Parallel model, Glover 2015 Table 2
    """
    por = 1 - bulkdens/partdens
    cond_wat = (rest_wat + 0.00001)**-1
    
    cond = (1 - por)*cond_matrix + vol_moist_cont*cond_wat + (por - vol_moist_cont)*cond_air
    return cond



def perpendicular(vol_moist_cont, bulkdens, partdens, cond_matrix, rest_wat, cond_air = 0.01e-3):
    """
    Perpendicular model, Glover 2015 Table 2
    """
    por = 1 - bulkdens/partdens
    cond_wat = (rest_wat + 0.00001)**-1
    
    cond = (((1-por)/cond_matrix) + vol_moist_cont/cond_wat + (por - vol_moist_cont)/cond_air)**-1
    return cond



def random(vol_moist_cont, bulkdens, partdens, cond_matrix, rest_wat, cond_air = 0.01e-3):
    """
    Random model, Glover 2015 Table 2
    """
    por = 1 - bulkdens/partdens
    cond_wat = (rest_wat + 0.00001)**-1
    
    cond = (cond_matrix**(1-por)) * (cond_wat**vol_moist_cont) * (cond_air**(por-vol_moist_cont))
    return cond


def HSme(vol_moist_cont, bulkdens, partdens, cond_matrix, rest_wat):
    """Hashin and Shtrikman (1962) Commonly denoted as HS-. Derived from effective medium considerations. Glover 2015 Table2"""
    por = 1 - bulkdens/partdens
    return cond_matrix*(1+(3*por*((1/rest_wat)-cond_matrix)/(3*cond_matrix+(1-por)*((1/rest_wat)-cond_matrix))))
       
                       
def HSma(vol_moist_cont, bulkdens, partdens, cond_matrix, rest_wat):
    """Hashin and Shtrikman (1962) Commonly denoted HS+. Derived from effective medium considerations. Glover 2015 Table2"""
    por = 1 - bulkdens/partdens
    return (1/rest_wat)*(1-(3*(1-por)*((1/rest_wat)-cond_matrix)/((3/rest_wat)-(por)*((1/rest_wat)-cond_matrix))))


def LRmodel(vmc, bulkdens, partdens, cond_matrix, rest_wat, m):    
    """Lichtenecker and Rother (1931), Korvin (1982) Derived from the theory of functional equations under appropriate
boundary conditions. Formally, the same as Archie’s law if s1=0. Glover 2015 Table 2"""
    por = 1 - bulkdens/partdens
    
    return (((1/rest_wat)**(1/m))*vmc + (cond_matrix**(1/m))*(1-por) )**m


def archie(bulk_dens, partdens, rest_wat, m):
    """
    Achie original formulation for saturated sandstones
    m: cementation exponent
    """
    por = 1 - (bulk_dens/partdens)
    return ((por**-m)*rest_wat)**-1


def comb_archie(vol_moist_cont, bulkdens, partdens, rest_wat, m, n):
    """ Archie first and second law combination"""
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por                                        # Saturation 
   
    return (por**(m))*(S**(n))*(1/rest_wat)
                       

def glover10(vmc, bulkdens, partdens, cond_matrix, rest_wat, m):
    """ Glover (2010b) Derived from the conventional Archie’s law by considering boundary conditions implied by geometric constraints. Tabla2, Glover 2015. Three phases"""
    
    por = 1 - bulkdens/partdens
    p = np.log(1-por**m)/np.log(1-por)
    #m2 = (vmc**m)/vmc
    m2 = np.log(1-vmc**m)/np.log(1-vmc)   # This exponent needs also to consider the exponent of the air phase (neglected)
    return (1/rest_wat)*vmc**m + cond_matrix*(1-por)**m2


def shahsingh(vol_moist_cont, clay_cont, rest_wat, c=1.45, m=1.25):
    """Shah and D. Singh, "Generalized Archie's Law for Estimation of Soil Electrical Conductivity"""
        
    if (clay_cont >= 5):
        c = .6 * clay_cont ** .55
        m = .92 * clay_cont ** .2
    return (c) * (1 / rest_wat) * vol_moist_cont ** m


def saey09H(clay_cont, organic_cont, LT):
    """ 
    Saey et al. 2009 (Ghent University) conductivity for horizontal coil horientation.
    S is set as unsaturated.
    Take a look to the original paper for details about how was measured clay_cont.
    
    LT = land use. This variable is dependent of organic_content, terefore must be set as = 1 if landuse = arable OR 
    = 0 if landuse = pasture. 
    """
    return (7.2 + 1.098 * clay_cont + 1.494 * LT * organic_cont + 1.339 * (1 - LT) * organic_cont)


def saey09V(clay_cont):
    """ 
    Saey et al. 2009 (Ghent University) conductivity for vertical coil horientation.
    S is set as unsaturated.
    Take a look to the original paper for details about how was measured clay_cont.
    """
    
    return (16.2 + 1.314 * clay_cont/100)


def peplinskicond(vol_moist_cont, bulkdens, partdens, clay_cont, sand_cont):
    """
    Empirical direct current electrical conductivity in imaginary component of complex permittivity of Peplinski 1995.  
    """
    alpha = 0.65
    
    beta = 1.33797 - 0.603*sand_cont/100 - 0.166*clay_cont/100
    condeff = 0.0467 + 0.2204*bulkdens - 0.4111*sand_cont/100 + 0.6614*clay_cont/100
    cond = ((condeff)*(partdens - bulkdens)*vol_moist_cont**(beta/alpha))/(partdens*vol_moist_cont)
    #cond = ((condeff)*(partdens - bulkdens))/(partdens*vol_moist_cont)
    #cond = condeff
    return cond


def rhoades90(vol_moist_cont, bulkdens, partdens, clay_cont, rest_wat):
    """
    Rhoades 1990 et al. 1b equation
    """
    cond_wat =   (1/rest_wat)
    thetas  =    bulkdens/partdens
    thetaws =    0.639*vol_moist_cont + 0.011 
    cond_solid = (0.023*clay_cont      - 0.021)
    condbulk =   (((thetas + thetaws)**2)/thetas)*cond_solid + (vol_moist_cont - thetaws)*cond_wat
    return condbulk


def waxsmidt68(vol_moist_cont, bulkdens, partdens, CEC, rest_wat, m, B0):
    """Waxman & Smidt 1968"""
    
    e = 2.718280828
    B = B0*( 1-0.6*e**(-1/(0.013*rest_wat)))
    por = 1 - bulkdens/partdens
    F = por**-m
    S = vol_moist_cont/por                                        # Saturation 
    Qv = partdens*((1-por)/por)*CEC  
    surfc = B*Qv*S/F
    #print("Surface conductivity Waxman&Smidt68", surfc)
    return (S**2)/(F*rest_wat)   +  surfc


def linde06(vol_moist_cont, bulkdens, partdens, CEC, rest_wat, m, n, B0):
    """
    Linde et al. 2006 conductivity equation. Surface conductivity proportion calculated as B*Qv    """
    
    e = 2.718280828
    B = B0*( 1-0.6*e**(-1/(0.013*rest_wat)))
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por                                        # Saturation 
    Qv = partdens*((1-por)/por)*CEC 
    cond_sup = B*Qv
    return (por**m) * ((S**n)* (1/rest_wat) + ((por**-m) - 1) * cond_sup)


def revil98(vol_moist_cont, bulkdens, partdens, CEC, rest_wat, m, n, B0, a=.4):
    """
    Revil et al 1998    """
    
    e = 2.718280828
    B = B0*( 1-0.6*e**(-1/(0.013*rest_wat)))
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por                                        # Saturation 
    Qv = partdens*((1-por)/por)*CEC                        # Excess of surface charge per unit volume
    
    return ((por**m)*(S**n)/a) * ((1/rest_wat)+(B*Qv/S))


def bardonpied(vmc, bulkdens, partdens, clay_cont, cond_clay, rest_wat, m):
    """
    Bardon and Pied 1969 in Table 4 Glover 2015"""
    
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    S = vmc/por
    F = por**-m
    return ((S**2)/(rest_wat*F)) + clay_cont*cond_clay*S


def schlumberger(vol_moist_cont, bulkdens, partdens, clay_cont, cond_clay, rest_wat, m):
    """ 
    schlumberger eq in Table 4 Glover 2015 """
    
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por
    F = por**-m
    return (S**2)/(F*(1-clay_cont)*rest_wat) + clay_cont*cond_clay*S


def hossin(vol_moist_cont, bulkdens, partdens, clay_cont, cond_clay, rest_wat, m):
    """ Hossin 1960 in Table 4 Glover 2015"""
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por
    F = por**-m
    return ((S**2)/(rest_wat*F)) + (clay_cont**2)*cond_clay


def juhasz(vol_moist_cont, bulkdens, partdens, clay_cont, cond_clay, rest_wat, m, partdens_clay=2.85):
    """ 
    Juhasz 1981 in Table 4 Glover 2015"""
    
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por
    F = por**-m
    por_clay = 1 - (bulkdens-0.2)/partdens_clay
    m_clay = np.log(cond_clay*rest_wat)/np.log(por_clay)
    F_clay = por_clay**-m_clay
    cond_wat = 1/rest_wat
    ##  POR ALGUN MOTIVO TIENE ALGUN PROBLEMAA 
    
    return ((S**2)/(rest_wat*F)) + ((-cond_clay/F_clay) + cond_wat)*((clay_cont*por_clay*S)/(por)) 
    
    
def wund13c(vol_moist_cont, L, rest_wat, rest_init, wat_init):  
    """
    #Wunderlich et.al 2013 Equation 22. This is trained fitting L, rest wat, wat_init and rest_init?    """
    dif = vol_moist_cont - wat_init                                       # Diference utilized just for simplicity
    y = rest_init**-1                                                     # Initial permitivity = Epsilon sub 1  
    x = 0                                                                 # Diferentiation from p = 0  
    dx = 0.001                                                            # Diferentiation step
    
    while x<1:                                                            # Diferentiation until p = 1
        dy = dx*((((y)*(dif))/(1-dif+x*dif)) * (((1/rest_wat)-(y))/((L/rest_wat) + (1-L)*y)))
        x=x+dx
        y=y+dy
    return y                                                              # Return electrical conductivity of the soil


def logseq7(cond, freq, freqc, n):
    """Equation 7 in Logsdon 2010 et al. for real electrical conductivity vs frequency"""
    return cond*(1 + freq/freqc)**n


def logs10MHz(vmc, cond, freq):
    """Equation 7 in Logsdon 2010 et al. for real electrical conductivity vs frequency
       'n' exponent and central frequency is calculated using Table 2 Logsdon 2005"""
    freqc = 8.53  - 25.4 *vmc + 0.782*cond*1000
    n =     0.682 + 0.371*vmc
    return  cond*(1 + ((freq)/(freqc*1000000)))**n
    
    
def clavier77(vmc, bd_eff, pd_eff, clay_cont, cc_cond, cc_bd, rest_wat, m, n, cc_pd = 2.8, a=1):
    """Clavier et al. (1977) dual-water model as defined in Glover etal. 2015 pag 20, eq. 45"""
    por_tsh = 1 - (cc_bd/cc_pd)
    #print("Clay porosity", por_tsh)
    por_eff = (1 - (bd_eff/pd_eff))*(1-clay_cont/100) + por_tsh*clay_cont/100
    #print("por_eff", por_eff)
    por_t =   por_eff + (clay_cont/100)*por_tsh
    #print("Total porosity", por_t)
    S_bw =    (clay_cont/100)*por_tsh/por_t                  # Bound water saturation
    #print("Bound water saturation", S_bw)
    ro_bw =   (por_tsh**2)/cc_cond                           # Bound water conductivity
    #print("Bound water rest", ro_bw)
    S_wt =    S_bw + (vmc/por_eff)
                                                
    return    ((S_wt**n) - S_wt*(S_bw*(1-(a*rest_wat/ro_bw)))) / (a*rest_wat*por_t**m)
    
    
def mcbratney05(vmc, bd, pd, clay, cec):
    """ Semiempirical model of McBratney et al. 2005 adjusted to Edgeroi soil database (n> 1900) for APARENT electrical conductivity.
    cec is in mmol/g"""
    k = 631.8   # mS/m 
    por = 1 - bd/pd
    cec0 = 1000 # mmol+/kg
    eca = k*(clay/100)*(vmc/por)*(cec*1000/cec0)
    return eca/1000

    
################################### ELECTRICAL CONDUCTIVITY VS TEMPERATURE ###################################


def mcneill80(t, cond25, beta=0.02):
    """
    Mcneill 1981 equation
    """
    return (cond25*(1+beta*(t-25)))/1000


def arps53(t, cond25):
    """
    Arps 1953 Equation
    """
    return (cond25*((t+21.5)/46.5))/1000


def luck09(t, cond25):
    """
    Luck 2009 Equation
    """
    return (cond25/(0.36+ e**((12.5-t)/28.5)))/1000


################################### ELECTRICAL CONDUCTIVITY OF WATER ###################################


def hilhorst_ecw(bulkcond, bulkperm, wp, offset):
    """ Hilhorst 2000 formula for water conductivity. It is given in the same utities as condbulk"""
    return (wp/bulkperm)*bulkcond + offset


def kelleners(vmc, bulkdens, partdens, waterdens, condsat):
    """ Unknown deduction process. It is given in the same utities as condsat"""
    por = 1 - bulkdens/partdens
    return (bulkdens/waterdens)*(vmc/por)*(condsat/vmc)


def vogeler96(vmc, bulkcond):
    """ EC-TDR based measuremets, Water conductivity was determined by extraction process """
    return (bulkcond - (0.228*vmc - 0.042))/(0.804*vmc-0.217)
    
    
def weast_KCl(C):
    """"For a KC1 solution with concentrations ranging from 0.005 to 0.05 M, Weast (1965, p. D-81"""
    return (C+0.7e-4)/(7.6e-2)


def weast_KCl(C):
    """"For a CaC12 solution with concentrations ranging from 0.005 to 0.05 M, Weast (1965, p. D-81"""
    return (C+1.2e-3)/(9.4e-2)
       
    
def kargas17(bulkcond, ap_perm, wp, offset=6):
    """Kargas et al 2017 (Prediction of Soil Solution Electrical Conductivity by the Permittivity Corrected Linear Model Using a Dielectric Sensor) using Robinson et al 99 and Hilhorst formula """
    return (wp/(((ap_perm**0.5) - 0.628*bulkcond)**2) - offset)*bulkcond
    
    
def leao(ap_perm, ap_cond, sand):
    """"New semi-empirical formulae for predicting soil solution conductivityfrom dielectric properties at 50 MHzTairone P. Leao et al. 2010"""
    #sand is expressed in percenaje
    sand = sand * 100
    return (ap_cond-0.08)/((ap_perm-6.2)*(0.0057+0.000071*sand))


def bouksila(ap_perm, ap_cond, wp):
    """Soil water content and salinity determination using different dielectric methods in saline gypsiferous soil / Détermination de la teneur en eau et de la salinité de sols salins gypseux à l'aide de différentes méthodes diélectriques FETHI BOUKSILA , MAGNUS PERSSON , RONNY BERNDTSSON & AKISSA BAHRI"""
    offset = 0.4414*(ap_cond/10)**3 - 4.3435*(ap_cond/10)**2 + 13.733*ap_cond/10 + 3.9181
    wp*ap_cond/(ap_perm - offset)
    
    
############################   MODIFIED MODELS FOR ELECTRICAL CONDUCTIVITY    ##########################
    
    
def modcomb_archie(vol_moist_cont, bulkdens, partdens, cc, rest_wat, perm_solid, n, offset):
    """ Archie first and second law combination. m calculed using prop 12"""
    
    alpha = (-0.46*cc/100)+0.71
    por = 1 - bulkdens/partdens
    m = (np.log((((1 - por)*(perm_solid/80)**alpha) + por)**(1/alpha))-offset/80) / np.log(por)
    S = vol_moist_cont/por                                      # Saturation 
    
    return (por**(m))*(S**(n))*(1/rest_wat)


def mod2comb_archie(vol_moist_cont, bulkdens, partdens, cc, rest_wat, perm_solid, wp, offset):
    """ Archie first and second law combination. m calculed using prop 12"""
    
    alpha = (-0.46*cc/100)+0.71
    por = 1 - bulkdens/partdens
    m = (np.log((((1 - por)*(perm_solid/wp)**alpha) + por)**(1/alpha))-offset/wp) / np.log(por)
    S = vol_moist_cont/por                                      # Saturation 
    n = m
    return (por**(m))*(S**(n))*(1/rest_wat)


def toppmod1(vol_moist_cont, rest_wat, wp):
    "Modified Topp equation through Brovelli and Cassiani 2008 procedure. This did not work well"
    p = [4.3e-6*(wp**3)*(rest_wat**3), 
         -5.5e-4*(wp**2)*(rest_wat**2), 
         2.92e-2*(wp)*(rest_wat),
         -5.3e-2 - vol_moist_cont]
    
    roots = np.roots(p) 
    roots = roots[roots.imag == 0 ]
    cond = roots[roots > 0]
    
    return cond[0].real
    
    
def toppmod2(vol_moist_cont, rest_wat, wp, offset = 4):
    "Modified Topp equation through Hillhost 2000 formula. This did not work well"
    a = -5.3e-2
    b = 2.92e-2 
    c = -5.5e-4
    d = 4.3e-6
    p = [d*(wp**3)*(rest_wat**3), 
         2*d*(wp**2)*(rest_wat**2) + d*offset*(wp**2)*(rest_wat**2) + c*(wp**2)*(rest_wat**2), 
         2*d*(wp)*(rest_wat)*offset + d*(wp)*(rest_wat)*offset**2 + b*(wp)*(rest_wat) + 2*c*offset*(wp)*(rest_wat),
         d*offset**3 + offset**2 + b*offset + a - vol_moist_cont]
    
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    cond = roots[roots > 0]
    return cond[0].real


def modlinde06(vol_moist_cont, bulkdens, pd, clay_cont, CEC, rest_wat, B0):
    """Linde et al. 2006 conductivity equation. Particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below), the effect seems negligible. 
    """
    if [clay_cont >= 5]:
        m = 0.92*clay_cont**0.2
    else:
        m = 1.25
    n = m    
    por = 1 - bulkdens/pd
    F =    por**-m
    e    = 2.718280828
    B    = B0*( 1-0.6*e**(-1/(0.013*rest_wat)))
    Qv   = pd*((1-por)/por)*CEC 
    S = vol_moist_cont/(por)
    
    return  (por**m) * (S**n)* (1/rest_wat) + (1-(por**m))*B*Qv


def mod2linde06(vmc, bulkdens, clay_cont, org_matter, rest_wat, air_perm, perm_solid, wp, n, cond_sup = .7, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """Linde et al. 2006 conductivity equation. Particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below), the effect seems negligible. 'm' is calculated using prop13 with a fixed offset==3. Surface conductivity is also fixed as 0.7 m S/m"""
    clay     = clay_cont/100
    org      = org_matter/100
    somr     = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd       = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1

    por      = 1 - bulkdens/pd
    alpha    = (-0.46 * clay) + 0.71
    S        = vmc/(por)
    offset   = 3                                   
    rate     = perm_solid / wp
    m        = np.log(((((1-por)*rate**alpha) + vmc + ((por-vmc)*(1/wp)**alpha))**(1/alpha)- (offset/wp))*S**(-n))/ np.log(por)
    return     (por**m) * ((S**n)* (1/rest_wat) + ((por**-m) - 1) * cond_sup/1000)


def mod3linde06(vmc, bulkdens, clay_cont, org_matter, rest_wat, air_perm, perm_solid, wp, n, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """     Linde et al. 2006 conductivity equation. Particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below), the effect seems negligible. 'm' is calculated using prop13 with a fixed offset==4. Surface
     conductivity is also fixed as 1 m S/m 
     """
    clay     = clay_cont/100
    org      = org_matter/100
    somr     = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd       = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
    
    por      = 1 - bulkdens/pd
    alpha    = (-0.46 * clay) + 0.71
    S        = vmc/(por)
    offset   = 4
    rate     = perm_solid / wp
    m        = np.log(((((1-por)*rate**alpha) + vmc + ((por-vmc)*(1/wp)**alpha))**(1/alpha)- (offset/wp))*S**(-n))/ np.log(por)
    #print("Linde 3 m=", m)
    cond_sup = 1/1000
    #print("Mod 3 Linde cond sup=", cond_sup)
    return     (por**m) * ((S**n)* (1/rest_wat) + ((por**-m) - 1)*cond_sup)


def mod4linde06(vmc, bulkdens, pd, clay, rest_wat):
    """Linde et al. 2006 conductivity equation. Particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below), the effect seems negligible. 'm' is calculated using prop12. Surface conductivity is calculated using 0.7/1000
    """
    por      = 1 - (bulkdens/pd)    
    alpha    = (-0.46 * clay/100) + 0.71
    S        = vmc/(por)
    m        = 1/alpha
    cond_sup = 0.7/1000
    n = m

    return     (por**m) * ((S**n)* (1/rest_wat) + ((por**-m) - 1)*cond_sup)


def mod5linde06(vmc, bulkdens, pd, clay_cont, rest_wat, n, cond_sup = .7):
    """Linde et al. 2006 conductivity equation. Particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below), the effect seems negligible. 'm' is calculated using prop13 with a fixed offset==3. Surface conductivity is also fixed as 0.7 m S/m"""
    por      = 1 - bulkdens/pd
    alpha    = (-0.46 * clay_cont/100) + 0.71
    S        = vmc/(por)
    m        = 1/alpha
    return     (por**m) * ((S**n)* (1/rest_wat) + ((por**-m) - 1) * cond_sup/1000)


def wundmod(vol_moist_cont, bulkdens, clay_cont, org_matter, rest_wat, air_perm, perm_solid, wp, rest_init = 1389, vol_wat_cont_init = 0.07, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):  
    
    """
    Wunderlich et.al 2013 Equation with a modification in the way which 'L' is introduced. Here is calculated using L = (-1/m)+1, where 'm' is calculated using prop12. This modifications does not seem realible cause in Wunderlich model 'L' refers to the shape of pores, not the particles as refers the relationship L = (-1/m)+1.    
    """
    clay = clay_cont/100
    org = org_matter/100
    somr = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
    
    por = 1 - bulkdens/pd
    alpha = (-0.46 * clay) + 0.71
    moist = vol_moist_cont
    #m = np.log((((1 - por)*(perm_solid/wp)**alpha) + por)) / (np.log(por)*alpha)
    offset = 4
    m = (np.log((((1 - por)*(5/80)**alpha) + por)**(1/alpha))-offset/80) / np.log(por)
    L = (-1/m) + 1
    
    dif = moist - vol_wat_cont_init                                       # Diference utilized just for simplicity
                                                                          # Initializing diferenciation parameters
    y = rest_init**-1                                                     # Initial permitivity = Epsilon sub 1  
    x = 0                                                                 # Diferentiation from p = 0  
    dx = 0.001                                                            # Diferentiation step
    
    while x<1:                                                            # Diferentiation until p = 1
        dy = dx*((((y)*(dif))/(1-dif+x*dif)) * (((1/rest_wat)-(y))/((L/rest_wat) + (1-L)*y)))
        x=x+dx
        y=y+dy
                                                                          # Return electrical conductivity of the soil
    return y


def modwaxmansmits(vol_moist_cont, bulkdens, clay_cont, org_matter, CEC, rest_wat, air_perm, perm_solid, wp, n, orgdens = 1.3, B=7.7e-8, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """
    Maxman and Smits 1968 model. Particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below). 'm' is calculated usinf prop12 with a fixed offset=4 and alpha with the empirical relationship  linkig it with clay content. Exponent for saturation is "n", no simply 2"
    """
    clay = clay_cont/100
    org = org_matter/100
    somr = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
    por = 1 - bulkdens/pd
    alpha = (-0.46 * clay) + 0.71
    S = vol_moist_cont/por  
    offset = 4
    m = (np.log((((1 - por)*(5/80)**alpha) + por)**(1/alpha))-offset/80) / np.log(por)
    Qv = pd*((1-por)/por)*CEC                        # Excess of surface charge per unit volume
    
    return ((por**m)*(S**n)/rest_wat) + ((por**m)*B*Qv/S)


def mod2waxsmidt(vol_moist_cont, bulkdens, partdens, CEC, rest_wat, m, n, B0):
    """Waxman & Smidt 1968 modification.
       Exponent for saturation is "n", no simply 2"""
    
    e = 2.718280828
    B = B0*( 1-0.6*e**(-1/(0.013*rest_wat)))
    por = 1 - bulkdens/partdens
    F = por**-m
    S = vol_moist_cont/por                                        # Saturation 
    Qv = partdens*((1-por)/por)*CEC  
            
    return (S**n)/(F*rest_wat)   +  B*Qv*S/F


def mod3waxsmidt(vmc, bulkdens, clay_cont, org_matter, CEC, rest_wat, air_perm, perm_solid, wp, n, offset, B0, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """
    Maxman and Smits 1968 model. Particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below). 'm' is calculated usinf prop9 where alpha with the empirical relationship  linkig it with clay content. Exponent for saturation is "n", no simply 2"
    """
    clay     = clay_cont/100
    org      = org_matter/100
    somr     = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd       = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
    
    por      = 1 - bulkdens/pd
    alpha    = (-0.46 * clay) + 0.71
    S        = vmc/(por)
    rate     = perm_solid / wp
    m        = 1/alpha
    
    e = 2.718280828
    B = B0*( 1-0.6*e**(-1/(0.013*rest_wat)))
    por = 1 - bulkdens/pd
    F = por**-m
    S = vmc/por                                        # Saturation 
    Qv = pd*((1-por)/por)*CEC  
    
    return     (por**m) * ((S**n)* (1/rest_wat)) + ((B*Qv)/(F))


def mod4waxsmidt(vmc, bulkdens, clay_cont, org_matter, CEC, rest_wat, air_perm, wp, m, n, B0, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """Maxman and Smits 1968 model. Particle density as function of organic matter and clay content as is described in Schjonning et al  2012 (def schjonnpd below). Exponent for saturation is "n", no simply 2
    """
    clay     = clay_cont/100
    org      = org_matter/100
    somr     = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd       = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
 
    por      = 1 - bulkdens/pd
    S        = vmc/(por)
    
    e        = 2.718280828
    B        = B0*( 1-0.6*e**(-1/(0.013*rest_wat)))
    por      = 1 - bulkdens/pd
    F        = por**-m
    S        = vmc/por                                        # Saturation 
    Qv       = pd*((1-por)/por)*CEC  
    
    return     (por**m) * ((S**n)* (1/rest_wat)) + ((B*Qv)/(F))


def modglover10(vmc, bulkdens, partdens, clay_cont, cond_matrix, rest_wat, perm_solid, wp, n, offset):
    """ Glover (2010b) Derived from the conventional Archie’s law by considering boundary conditions implied by geometric constraints.
    Tabla2, Glover 2015. Three phases. The modification is introduced through 'm' using prop9
    """
    
    por =    1 - bulkdens/partdens
    alpha = -0.46*clay_cont/100 + 0.71 
    S =     vmc/por 
    rate =  perm_solid/wp
    m =     1/alpha
    m2 =    np.log(1-vmc**m)/np.log(1-vmc)   # This exponent needs also to consider the exponent of the air phase (neglected)
    return  (1/rest_wat)*vmc**m + cond_matrix*(1-por)**m2


def LRMod1(vmc, bulkdens, partdens, clay_cont, cond_matrix, rest_wat):    
    """Lichtenecker and Rother (1931), Korvin (1982) Derived from the theory of functional equations under appropriate
boundary conditions. Formally, the same as Archie’s law if s1=0. Glover 2015 Table 2. 'm' is calculated using prop9"""
    por =   1 - bulkdens/partdens
    alpha = -0.46 * clay_cont/100 + 0.71
    m =     1/ alpha
    return  (((1/rest_wat)**(1/m))*vmc + (cond_matrix**(1/m))*(1-por) )**m


def LRMod2(vmc, bulkdens, partdens, clay_cont, cond_matrix, rest_wat, perm_solid, wp):    
    """Lichtenecker and Rother (1931), Korvin (1982) Derived from the theory of functional equations under appropriate
boundary conditions. Formally, the same as Archie’s law if s1=0. Glover 2015 Table 2. 'm' is calculated using prop11"""
    rate  = perm_solid/wp
    alpha = -0.46 * clay_cont/100 + 0.71
    por   = 1 - bulkdens/partdens              
    m     = np.log(((1 - por)*(rate)**alpha) + por) / (np.log(por)*alpha)
    
    return (((1/rest_wat)**(1/m))*vmc + (cond_matrix**(1/m))*(1-por) )**m


def LRMod3(vmc, bulkdens, partdens, clay_cont, cond_matrix, rest_wat):    
    """Lichtenecker and Rother (1931), Korvin (1982) Derived from the theory of functional equations under appropriate
boundary conditions. Formally, the same as Archie’s law if s1=0. Glover 2015 Table 2.  'm' is calculated using prop13"""
    por =    1 - bulkdens/partdens
    S      = vmc/por
    
    if [clay_cont >= 5]:
        m = 0.92*clay_cont**0.2
    else:
        m = 1.25
        
    return   (((1/rest_wat)**(1/m))*vmc + (cond_matrix**(1/m))*(1-por) )**m


def bardonpiedMod1(vmc, bulkdens, partdens, clay_cont, cond_clay, rest_wat, perm_solid, n):
    """
    Modification of Bardon and Pied 1969 in Table 4 Glover 2015
    'm' is calculated as in prop11 and the exponent of S is 'n' not just 2"""
    
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    alpha = -0.46 * clay_cont/100 + 0.71
    rate  = perm_solid/80
    offset= 4
    m     = (np.log((((1 - por)*(rate)**alpha) + por)**(1/alpha) - (offset/80))) / np.log(por)
    S     = vmc/por
    F     = por**-m
    return ((S**2)/(rest_wat*F)) + clay_cont*cond_clay*S


def schlumberger1(vol_moist_cont, bulkdens, partdens, clay_cont, cond_clay, rest_wat):
    """ 
    Modification 1 of schlumberger eq in Table 4 Glover 2015 """
    
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por
    
    if [clay_cont >= 5]:
        m = 0.92*clay_cont**0.2
    else:
        m = 1.25
        
    F = por**-m
    n = m
    return (S**2)/(F*(1-clay_cont)*rest_wat) + clay_cont*cond_clay*S


def schlumberger2(vol_moist_cont, bulkdens, partdens, clay_cont, cond_clay, rest_wat, ps, wp, offset):
    """ 
    Modification 2 of schlumberger eq in Table 4 Glover 2015 """
    
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por
    rate = ps / wp
    alpha = -0.46 * clay_cont/100 + 0.71
    m = (np.log((((1 - por)*(rate)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    F = por**-m
    return (S**2)/(F*(1-clay_cont)*rest_wat) + clay_cont*cond_clay*S


def schlumberger3(vol_moist_cont, bulkdens, partdens, clay_cont, cond_clay, rest_wat):
    """ 
    Modification 3 of schlumberger eq in Table 4 Glover 2015 """
    
    clay_cont = clay_cont/100
    por = 1 - bulkdens/partdens
    S = vol_moist_cont/por
    alpha = -0.46 * clay_cont/100 + 0.71
    m = 1/alpha
    F = por**-m
    return (S**2)/(F*(1-clay_cont)*rest_wat) + clay_cont*cond_clay*S


 ###############################################    IMAGINARY ELECTRICAL CONDUCTIVITY  ####################################


def logsdon_eq9i(fc, rperm, iperm):
    """
    Logsdon et al 2010 eq 8 for imaginary conductivity"""
    
    pi = 3.141592653589793
    e_0 = 8.8541878028e-12
    arc = np.arctan(2.0*iperm/rperm)
    return arc*iperm*pi*fc*e_0

    
 ###############################################    Real PERMITIVITY EQUATIONS   ####################################


def MG_2phase(perm_solid, air_perm, bulkdens, partdens):
    """
    The Maxwell-Garnett [1904] mixing model based on the Lord Rayleigh [1892] formula is the most commonly used model for describing a twophase mixture (Robinson & Friedman 2003)"""
    por = 1 - bulkdens/partdens
    f =   1 - por
    return air_perm + 3*f*air_perm*((perm_solid - air_perm)/(perm_solid + 2*air_perm - f*(perm_solid - air_perm)))


def bosch_silt(vmc):
    """Comparison of Capacitance-Based Soil Water Probes in Coastal Plain Soils. David D. Bosch. 2004. bulk density 1.54. sand 87 silt 7 clay 6. Bottom leaching drying experiment. Real permittivity vs moisture content using HydraProbe"""
    p = [ 1.632e-5, -9.751e-4, 3.251e-2, -0.0863 - vmc ]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    return (perm_rel[0].real)


def bosch_sand(vmc):
    """Comparison of Capacitance-Based Soil Water Probes in Coastal Plain Soils. David D. Bosch. 2004. bulk density 1.57. sand 90 silt 7 clay 3. Bottom leaching drying experiment. Real permittivity vs moisture content using HydraProbe"""
    p = [ 7.587e-6, -9.331e-4, 3.861e-2, -0.1304 - vmc ]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    return (perm_rel[0].real)
    
    
def bosch_clay(vmc):
    """Comparison of Capacitance-Based Soil Water Probes in Coastal Plain Soils. David D. Bosch. 2004. bulk density 1.51. sand 60 silt 9 clay 31. Bottom leaching drying experiment. Real permittivity vs moisture content using HydraProbe"""
    p = [ 3.350e-5, -2.519e-3, 6.625e-2, -0.2093 - vmc ]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    return (perm_rel[0].real)
    
    
def ojo_MB1(perm):
    """ Laboratory calibrated equation for sandy soil using Hydraprobe. Calibration and Evaluation of a Frequency Domain
Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return (0.1127* perm**0.5)-0.2025


def ojo_MB4(perm):
    """ Laboratory calibrated equation for sandy soil using Hydraprobe. Calibration and Evaluation of a Frequency Domain
Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return perm*(1.085e-2) - (1.2122e-2)*perm**(0.5) + 3.082e-2


def ojo_MB6(perm):
    """ Laboratory calibrated equation for clay soil using Hydraprobe. Calibration and Evaluation of a Frequency Domain
Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return  (6.090e-3)*perm + (2.0246e-2)*perm**0.5 + 9.746e-3


def ojo_MB7(perm):
    """ Laboratory calibrated equation for sandy soil using Hydraprobe. Calibration and Evaluation of a Frequency Domain
Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return (0.1084* perm**0.5)-0.1949


def ojo_MB8(perm):
    """ Laboratory calibrated equation for clay soil using Hydraprobe. Calibration and Evaluation of a Frequency Domain
Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return  8.066e-3*perm + 8.76e-4*perm**0.5 + 3.9367e-2


def ojo_MB9(perm):
    """ Laboratory calibrated equation for sandy soil using Hydraprobe. Calibration and Evaluation of a Frequency Domain
Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return (0.1131* perm**0.5)-0.2116


def ojo_all(perm):
    """ Laboratory calibrated equation for all the soils using Hydraprobe. Calibration and Evaluation of a Frequency Domain
Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return 0.0870*perm**0.5 - 0.1425


def ojo_15cec(rperm):
    """ FIELD calibrated equation for soils with less than 15 meq/100gr CEC using Hydraprobe (Table 5, R2=0.94). Calibration and Evaluation of a Frequency Domain Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return (0.1084*rperm**(0.5))-0.1633
    
    
def ojo_15_30cec(rperm):
    """ FIELD calibrated equation for soils with 15 to 30 meq/100gr CEC using Hydraprobe (Table 5, R2=0.87). Calibration and Evaluation of a Frequency Domain Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return (0.0786*rperm**(0.5)) - 0.0714


def ojo_30cec(rperm):
    """ FIELD calibrated equation for soils with more than 30 meq/100gr CEC using Hydraprobe (Table 5, R2=0.77). Calibration and Evaluation of a Frequency Domain Reflectometry Sensor for Real-Time Soil Moisture Monitoring E. RoTimi Ojo"""
    return (3.32*10**-3)*rperm**(1.5) - (6.784*10**-2)*rperm + (5.047*10**-1)*rperm**(0.5) - 8.85*10**-1


def Hydraprobe(vol_moist_cont):
    """
    Hydraprobe default equation for VMC (eauqtion A2, apendix C)
    """
    A = 0.109
    B = -0.179
    
    return (((vol_moist_cont - B)/A)**2)*e_0
    
    
def topp1980(vol_moist_cont):
    """
    Topp et al. (1980). The permittivity is the aparent one.
    """
    p = [4.3e-6, -5.5e-4, 2.92e-2, -5.3e-2 - vol_moist_cont]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)*e_0


def logsdonperm(vol_moist_cont):
    'Logsdon 2010 eq10'
    
    p = [0.00000514, -0.00047, 0.022, -vol_moist_cont]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)*e_0
    
    
def nadler1991(vol_moist_cont):
    """
    Nadler et al. (1991)
    """
    p = [15e-6, -12.3e-4, 3.67e-2, -7.25e-2 - vol_moist_cont]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)*e_0


def roth1992(vol_moist_cont):
    """
    Roth et al. (1992) 
    """
    p = [36.1e-6, -19.5e-4, 4.48e-2, -7.28e-2 - vol_moist_cont]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    
    return (perm_rel[0].real)*e_0


def jacandschj1993A(vol_moist_cont):
    """
    Jacobsen and Schjonning (1993) Equation (1)
    """
    p = [18e-6, -11.6e-4, 3.47e-2, -7.01e-2 - vol_moist_cont]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    return (perm_rel[0].real)*e_0


def jacandschj1993B(vol_moist_cont, bulkdens, clay_cont, organic_cont):
    """
    #Jacobsen and Schjonning (1993) Equation (2)
    """
    p = [17.1e-6, -11.4e-4, 3.45e-2, 
         -3.41e-2 - vol_moist_cont -3.7e-2 * bulkdens + 7.36e-4 * clay_cont + 47.7e-4 * organic_cont]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    return (perm_rel[0].real)*e_0


def hallikainen_1_4(vmc, clay, sand):
    """ Empirical model for real permittivity at 1.4 Ghz. Microwave Dielectric Behavior of Wet Soil-Part 1: Empirical Models and Experimental Observations MARTTI T. HALLIKAINEN 1985  """
    a0 = 2.862 
    a1 = -0.012 
    a2 = 0.001 
    b0 = 3.803 
    b1 = 0.462 
    b2 = -0.341 
    c0 = 119.006 
    c1 = -0.500 
    c2 = 0.633 
    return ((a0 + a1*sand + a2*clay) + (b0 + b1*sand + b2*clay)*vmc + (c0 + c1*sand + c2*clay)*vmc**2)*e_0


def hallikainen_4(vmc, clay, sand):
    """ Empirical model for real permittivity at 4 Ghz. Microwave Dielectric Behavior of Wet Soil-Part 1: Empirical Models and Experimental Observations MARTTI T. HALLIKAINEN 1985 """
    a0 = 2.927
    a1 = -0.012 
    a2 = -0.001 
    b0 = 5.505 
    b1 = 0.371 
    b2 = 0.062
    c0 = 114.826
    c1 = -0.389
    c2 = -0.547
    return ((a0 + a1*sand + a2*clay) + (b0 + b1*sand + b2*clay)*vmc + (c0 + c1*sand + c2*clay)*vmc**2)*e_0

    
def hallikainen_18(vmc, clay, sand):
    """ Empirical model for real permittivity at 18 Ghz. Microwave Dielectric Behavior of Wet Soil-Part 1: Empirical Models and Experimental Observations MARTTI T. HALLIKAINEN 1985 """
    a0 = 1.912
    a1 = 0.007
    a2 = 0.021
    b0 = 29.123 
    b1 = -0.190 
    b2 = -0.545
    c0 = 6.960
    c1 = 0.822
    c2 = 1.195
    return ((a0 + a1*sand + a2*clay) + (b0 + b1*sand + b2*clay)*vmc + (c0 + c1*sand + c2*clay)*vmc**2)*e_0


def raizfunc(vol_moist_cont):

    return(((vol_moist_cont + 0.1788)/0.1138)**2)*e_0 


def steelman(vol_moist_cont):
    """
    Colby M. Steelman* and Anthony L. Endres (2011) 
    """
    
    p = [2.97e-5, -2.03e-3, 5.65e-2, -0.157 - vol_moist_cont]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    perm_rel = roots[roots > 0]
    return (perm_rel[0].real)*e_0


def malicki(vol_moist_cont, bulkdens):
    """
    Malicki et al. 1996 
    """
    return((vol_moist_cont*(7.17 + 1.18*bulkdens) + 0.809 + 0.168*bulkdens + 0.159*bulkdens**2)**2)*e_0
    
    
def CRIM(vmc, bulkdens, partdens, air_perm, perm_solid, wp, alpha = 0.5):
    """
    #CRIM Equation (Birchak et.al 1974)
    """
    por = 1 - bulkdens/partdens
    return (( vmc*wp**alpha + (1-por)*perm_solid**alpha + (por-vmc)*air_perm**(alpha))**(1/alpha))*e_0


def roth1990(vmc, bulkdens, partdens, air_perm, perm_solid, wp, alpha): 
    """
    # Roth et.al (1990) Equation
    """
    por = 1 - bulkdens/partdens                          
    return (( vmc*wp**alpha + (1-por)*perm_solid**alpha + (por-vmc)*air_perm**(alpha))**(1/alpha))*e_0


def sen81eq23(bulkdens, partdens, perm_solid, wp):
    """ Sen 1981 equation 23 for saturated conditions and spherical grains, in DC limit """
    por = 1 - bulkdens/partdens
    return (1.5*perm_solid + (por**1.5)*(wp - 1.5*perm_solid))*e_0


def pride94(vol_moist_cont, bulkdens, partdens, perm_solid, wp, m):
    """
    Pride 1994 Equation for permittivity (eq. 253)
    """
    por = 1 - bulkdens/partdens
    
    return ((por**m) * (wp - perm_solid) + perm_solid)*e_0


def lindeperm(vol_moist_cont, bulkdens, partdens, air_perm, perm_solid, wp, m, n):
    """
    Linde permittivity 2006
    """
    por = 1 - bulkdens/partdens
    S = vol_moist_cont / por
    
    return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*perm_solid))*e_0


def cornelis(vol_moist_cont, bulkdens, partdens, perm_solid):
    """
    Equation shown in Corneli's course
    """
    return ((1 + (((perm_solid**0.5) - 1) *  bulkdens)/partdens + 8*vol_moist_cont)**2)*e_0


def dobson85(vmc, bound_wat, bulkdens, partdens, air_perm, perm_solid, wp, bwp):
    """"Microwave Dielectric Behavior of Wet Soil-Part II: Dielectric Mixing Models MYRON C. DOBSON, MEMBER, IEEE, FAWWAZ T. ULABY, FELLOW, IEEE, MARTTI T. HALLIKAINEN, MEMBER, IEEE, AND MOHAMED A. EL-RAYES (1.4 and 18 GHz), eq 21, r = .98 """
    por = 1 - bulkdens/partdens
    num = 3*perm_solid+2*(vmc - bound_wat)*(wp-perm_solid) + 2*bound_wat*(bwp-perm_solid)+2*(por-vmc)*(air_perm-perm_solid)
    den = 3+(vmc-bound_wat)*((perm_solid/wp)-1)+bound_wat*((perm_solid/bwp)-1)+(por-vmc)*((perm_solid/air_perm)-1)
    return (num/den)*e_0


def sen1981(vol_moist_cont, bulkdens, partdens, L, air_perm, perm_solid, wp):        
    """
    # Sen et.al 1981 Equation
    """
    por = 1 - bulkdens/partdens
    cl = vol_moist_cont*(L*perm_solid + (1-L)*wp)          # Calculation just for simplicity 
    wcg = wp*(((1-por)*perm_solid+cl) / ((1-por)*wp+cl))   # wcg = wat coated grains
    df = (por*-1) + vol_moist_cont                         # Diference utilized just for simplicity
                                                           # Initializing diferenciation parameters
    y = air_perm                                           # Initial permitivity = Epsilon sub a  
    x = 0.001                                              # Diferentiation from p = 0  
    dx = 0.01       
                                                           # Diferentiation step
    while x<1:                                             # Diferentiation until p = 1
        dy = ((y*(1+df))/(-df+x*(1+df))) * ((wcg-y)/(L*wcg+(1-L)*y))
        x=x+dx
        y=y+dy*dx
        
    return y*e_0 
 
    
def fengsen1985(vol_moist_cont, bulkdens, partdens, L, air_perm, perm_solid, wp):
    """
    # Feng & Sen 1985 Equation
    """
    por = 1 - bulkdens/partdens
    vmc = vol_moist_cont                                   # Abreviation just for simplicity 
    pds = perm_solid                                       # Abreviation just for simplicity
                                                           # Initializing diferenciation parameters
    y = wp                                                 # Initial permitivity = Epsilon sub a  
    x = 0                                                  # Diferentiation from p = 0  
    dx = 0.001                                             # Diferentiation step
    
    while x<1:                                             # Diferentiation until p = 1
        dy = (y/(vmc+x*(1-vmc))) * ((((1-por)*((pds-y))/(L*pds+(1-L)*y))) + ((por-vmc)*(air_perm-y))/(L*air_perm+(1-L)*y)) 
        x = x + dx
        y = y + dy*dx
    return y*e_0 



def endresredman1996(vol_moist_cont, bulkdens, partdens, L, air_perm, perm_solid, wp):   
    """
    # Endres & Redman 1996 Equation
    """
    por = 1 - bulkdens/partdens
    vmc = vol_moist_cont                                          # Abreviation just for simplicity 
    pds = perm_solid                                              # Abreviation just for simplicity
    S = vmc/por                                                   # Saturation
                                                                  # Initializing diferenciation parameters
    y = wp                                                        # Initial permitivity = Epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.001                                                    # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((air_perm-y)/(L*air_perm+(1-L)*y))  
        x = x + dx
        y = y + dy
                                                                  # Now y is equal to permitivity of pore(s)
    p = 0
    dp = 0.001
    z = y
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((pds-z)/(L*pds+(1-L)*z))
        p = p + dp
        z = z + dz
    return z*e_0


def wunderlich2013perm(vol_moist_cont, L, wp, vol_wat_cont_init = 0.04, perm_init = 6): 
    """Wunderlich et.al 2013 Equation for permitivity.
       Consider the fact that clay content has an appreciable influence in wp. Pag 10/14 Wunderlich et al.2013
    """

 
    diff = vol_moist_cont - vol_wat_cont_init                      # Diference utilized just for simplicity
                                                                   # Initializing diferenciation parameters
    y = perm_init                                                  # Initial permitivity = Epsilon sub 1  
    x = 0.001                                                      # Diferentiation from p = 0  
    dx = 0.001                                                     # Diferentiation step
                                                                   # Diferentiation until p = 1
    while x<1:                                                    
        dy = ((y*diff)/(1-diff+x*diff)) * ((wp-y)/(L*wp+(1-L)*y))
        x=x+dx
        y=y+dy*dx
    return y*e_0


def peplinski(vol_moist_cont, bulkdens, partdens, clay_cont, sand_cont, perm_solid, frec, wp, ewinf, alpha = 0.65, tau = 0.58e-10):
    """
    Peplinski 1995 model for real component of permittivity eqs 1 - 6 
    """
    robulkrosolid = bulkdens/partdens
    beta1 = 1.2748 - 0.519*sand_cont/100 - 0.152*clay_cont/100
    efw = ewinf + (wp - ewinf)/(1 + (tau*frec)**2)
    bulkperm = (1 + robulkrosolid*perm_solid**alpha + (vol_moist_cont**beta1)*(efw**alpha) - vol_moist_cont)**(1/alpha)
    return bulkperm*e_0


def stratton41(bulkdens, partdens, L, perm_solid, wp):
    """
    Eq A-1 Sen et al. 1981
    """
    por = 1 - (bulkdens/partdens)
    bulkperm = wp*((wp+(perm_solid-wp)*(1-por+por*L))/(wp + por*L*(perm_solid-wp)))
    return bulkperm*e_0


def strattonL0(vmc, bulkdens, partdens, perm_solid, wp):
    """Sen 1981 Eq A-3 derived using L=0 in Stratton 1941 equation. It is valid for complex permittivity. 
    Porosity along with wp was replaced for moisture content."""
    
    por = 1- bulkdens/partdens
    return ((1 - por)*perm_solid + vmc*wp)*e_0


def strattonL1(vmc, bulkdens, partdens, perm_solid, wp):
    """Sen 1981 Eq A-4 derived using L=1 in Stratton 1941 equation. It is valid for complex permittivity.
    Porosity along with wp was replaced for moisture content."""
    por = 1- bulkdens/partdens
    return (((1 - por)/perm_solid + vmc/wp)**-1)*e_0


def logsdon(fc, rperm, iperm):
    """
    Equation 8 of Logsdon 2010
    """
    pi = 3.141592653589793
    e_0 = 8.8541878028e-12
    arc = np.arctan(2.0*iperm/rperm)
    return arc*rperm*pi*fc*e_0


def debye(frec, time, eps_0, eps_inf):
    """Debye model for frequency dependence of polarization process as defined in Glover 2015 pag 27. The perm utilized can be refered to a mean values for different moisture contents."""
    """ The value for ε∞ = 4.9, while εfws and τfw are calculated as a function of pore water salinity and temperature following Stogryn (1971)."""
    
    return eps_inf + (eps_0-eps_inf)/(1+((2*np.pi*frec)**2)*time**2)
    

def MWBH(    bd, pd, rsw, ps, freq, wp):
    """"Permittivity WMBH model for frequency between 100Mhx and 1GHz as described (eq 5) and tested in Chen, Y., and D. Or (2006). Here the samples are sands and sandy loam almost saturated, while debye is not."""
    e_0 =      8.8541878028e-12                                   # Vacum permittivity
    pi =       3.141592653589793
    por =      1 - bd/pd
    pors =     1 - por
    deltae =   (9**pors*(1-pors)*ps**2) / (((2+pors)**2)*(2*wp+ps+pors*(wp-ps)))
    fr =       (2+pors)/(rsw*2*np.pi*e_0*(2*wp+ps+pors*(wp-ps)))  # Relaxation frequency
    t =        1/(2*np.pi*fr)
    sigma =    2*(1-pors)/(rsw*(2+pors))
    w    =     2*np.pi*freq
    perm =     (wp*(2*(1-pors)*wp + (1+2*pors)*ps) / (wp*(2+pors)+ps*(1-pors))) + deltae/(1+ (w*t)**2) + sigma/(w*e_0)
    
    return perm
    
    
################################ REAL Water PERMITTIVITY VS TEMPERATURE #########################################


def jones05(T):
    """ Jones 2005 equation as defined in HydraProbe manual for water permittivity"""
    #T = T + 273.15
    perm = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    return perm


def handbook(T):
    """ Weast, R.C. 1986. CRC Handbook of Chemistry and Physics. CRC
soils. Applied Geophysics 49:73–88. Press, Boca Raton, FL."""
    return 78.5411- (4.579e-3)*(T-25)+ (1.19e-5)*(T-25)**2 - (2.8e-8)*(T-25)**3


def teros12pt(T):
    """ Water permittivity vs T as defined in eq 3 in Teros12 manual """
    return 80.3 - 0.37*(T-20)


##################################    REAL BULK PERMITTIVITY MODELS FOR Temperature #############################


# Bichark et al. (1974) shows a combination of volumetric mixing model where the temperature is introduced trough the water phase permittivity. This is the base of all the models described in this section. See for instance Seyfried & Murdock 2004 page 401.

def tCRIM(vol_moist_cont, bulkdens, partdens, perm_solid, T, air_perm = 1, alpha = 0.5):
    """
    #CRIM Equation (Birchak et.al 1974)
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    por = 1 - bulkdens/partdens
    return ((vol_moist_cont * ((wp**alpha) - (air_perm**alpha)) + ((1-por)*(perm_solid**alpha)) + por*(air_perm**alpha))**(1/alpha))


def troth1990(vol_moist_cont, bulkdens, partdens, perm_solid, T, air_perm = 1): 
    """
    # Roth et.al (1990) Equation
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    por = 1 - bulkdens/partdens
    alpha = 0.5                           
    return ((vol_moist_cont * ((wp**alpha) - (air_perm**alpha)) + ((1-por)*(perm_solid**alpha)) + por*(air_perm**alpha))**(1/alpha))


def tsen1981(vol_moist_cont, bulkdens, partdens, L, perm_solid, T, air_perm = 1):        
    """
    # Sen et.al 1981 Equation
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    por = 1 - bulkdens/partdens
    cl = vol_moist_cont*(L*perm_solid + (1-L)*wp)                            # Calculation just for simplicity 
    wcg = wp*(((1-por)*perm_solid+cl) / ((1-por)*wp+cl))   # wcg = wat coated grains
    df = (por*-1) + vol_moist_cont                                          
    y = air_perm                                                                 
    x = 0.001                                                                        # Diferentiation from p = 0  
    dx = 0.01                                                                        # Diferentiation step
    
    while x<1:                                                                       # Diferentiation until p = 1
        dy = ((y*(1+df))/(-df+x*(1+df))) * ((wcg-y)/(L*wcg+(1-L)*y))
        x=x+dx
        y=y+dy*dx
    return y
   
    
def tfengsen1985(vol_moist_cont, bulkdens, partdens, L, perm_solid, T, air_perm = 1):
    """
    # Feng & Sen 1985 Equation
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    por = 1 - bulkdens/partdens
    vmc = vol_moist_cont                                          # Abreviation just for simplicity 
    pds = perm_solid                                          # Abreviation just for simplicity
                                                                  # Initializing diferenciation parameters
    y = wp                                                # Initial permitivity = Epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.001                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = (y/(vmc+x*(1-vmc))) * ((((1-por)*((pds-y))/(L*pds+(1-L)*y))) + ((por-vmc)*(air_perm-y))/(L*air_perm+(1-L)*y)) 
        x = x + dx
        y = y + dy*dx
    return y


def tendresredman1996(vol_moist_cont, bulkdens, partdens, L, perm_solid, T, air_perm = 1):   
    """
    # Endres & Redman 1996 Equation
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    por = 1 - bulkdens/partdens
    vmc = vol_moist_cont                                          # Abreviation just for simplicity 
    pds = perm_solid                                          # Abreviation just for simplicity
    S = vmc/por                                              # Saturation
                                                                  # Initializing diferenciation parameters
    y = wp                                                # Initial permitivity = Epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.001                                                     # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = ((dx*y*(1-S))/(S+x*(1-S))) * ((air_perm-y)/(L*air_perm+(1-L)*y))  
        x = x + dx
        y = y + dy
                                                                  # Now y is equal to permitivity of pore(s)
    p = 0
    dp = 0.001
    z = y
    while p<1:    
        dz = (dp*z*(1-por))/(por+p*(1-por)) * ((pds-z)/(L*pds+(1-L)*z))
        p = p + dp
        z = z + dz
    return z


def twunderlich2013perm(vol_moist_cont, L, T, vol_wat_cont_init = 0.04, perm_init = 6): 
    """
    #Wunderlich et.al 2013 Equation for permitivity
    """
    # Taking into account the fact that clay content has an appreciable influence in wp. Pag 10/14 Wunderlich et al.2013
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    
    diff = vol_moist_cont - vol_wat_cont_init                      # Diference utilized just for simplicity
                                                                   #Initializing diferenciation parameters
    y = perm_init                                                  # Initial permitivity = Epsilon sub 1  
    x = 0.001                                                      # Diferentiation from p = 0  
    dx = 0.001                                                     # Diferentiation step
                                                                   # Diferentiation until p = 1
    while x<1:                                                    
        dy = ((y*diff)/(1-diff+x*diff)) * ((wp-y)/(L*wp+(1-L)*y))
        x=x+dx
        y=y+dy*dx
    return y


def tpride94(vol_moist_cont, bulkdens, partdens, L, perm_solid, n, T):
    """
    Pride 1994 Equation for permittivity (eq. 253)
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    por = 1 - bulkdens/partdens
    m = 1/(1-L)
    return ((por**m) * (wp - perm_solid) + perm_solid)


def tlindeperm(vol_moist_cont, bulkdens, partdens, L, perm_solid, n, T, air_perm = 1):
    """
    Linde permittivity 2006
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    por = 1 - bulkdens/partdens
    m = 1/(1-L)
    S = vol_moist_cont / por
    #The folowing equation takes intp account the influence of air in bulk permittivity, the changes are minimum
    #return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*perm_solid + (1 - S**n)*air_perm))*e_0
    return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*perm_solid))


def tpeplinski(vol_moist_cont, bulkdens, partdens, clay_cont, sand_cont, perm_solid, frec, T, ewinf, alpha = 0.65, tau = 0.58e-10):
    """
    Peplinski 1995 model for real component of permittivity eqs 1 - 6 
    """
    wp = 80*(1-(4.579e-3) *(T-25)+ (1.19e-5)* ((T-25)**2) - (2.8e-8)*(T-25)**3)
    robulkrosolid = bulkdens/partdens
    beta1 = 1.2748 - 0.519*sand_cont/100 - 0.152*clay_cont/100
    efw = ewinf + (wp - ewinf)/(1 + (tau*frec)**2)
    bulkperm = (1 + robulkrosolid*perm_solid**alpha + (vol_moist_cont**beta1)*(efw**alpha) - vol_moist_cont)**(1/alpha)
    return bulkperm


##########################################   SOLID PHASE PERMITTIVITY   ###########################################


def CRIM_es(bulkperm, air_perm, bulkdens, partdens):
    """
    CRIM model for solid phase permittivity"""
    por = 1 - bulkdens/partdens
    return ((bulkperm**0.5 - por*air_perm**0.5)/(1-por))**2


def linde_es(bulkperm, bulkdens, partdens, m):
    """
    Linde et al. 2006 model for solid phase permittivity"""
    por = 1 - bulkdens/partdens
    F = por**-m
    return (F/(F-1))*bulkperm


def olhodft(bulkperm, bulkdens, partdens):
    """
    'Olhoeft [1981] used the expression (1a) based on the Lichtenecker [1926] equation (1b), which simply averages the logarithms of the permittivities' (Robinson & Friedman 2003)"""
    por = 1 - bulkdens/partdens
    return bulkperm**(1/(1-por))
    
    
def nelson(bulkperm, bulkdens, partdens):
    """
    'Nelson et al. [1989] favored the Looyenga [1965] mixing formula' (Robinson & Friedman 2003)"""
    por = 1 - bulkdens/partdens
    f = 1 - por
    return ((bulkperm**0.3 + f - 1)/f)**3


def bruggeman(bulkperm, air_perm, bulkdens, partdens):
    """
    Bruggeman formula, the symmetric effective medium approximation [Bruggeman, 1935] (Robinson & Friedman 2003)"""
    por = 1 - bulkdens/partdens
    F =   (-por*(air_perm**0.5)) + 1
    return (1 - F - bulkperm**(2/3)) / (1 - F - bulkperm**(-1/3))


def Dobson(part_dens):
    """" Ca is 1.0 and es is determined by an empirical fitting of the data presented in Part I for soils having extremely low moisture contents. The resultant expression es = (1.01 + 0.44 p,)2 - 0.062 (22) (at frequencies between 1.4 and 18 GHz.) yields es 4.7 at the specific densities given in Table 1. Equation(22) is nearly identical to that obtained by Shutko [21] and Krotikov [22] for other soils.(Microwave Dielectric Behavior of Wet Soil-Part II: Dielectric Mixing Models, Dobson 1985)"""
    return (1.01+0.44*part_dens)**2-0.062


def HBS_es(bulkdens, partdens, bulk_perm, air_perm):
    "Bruggeman 1935; Hanai 1968; Sen et al. 1981 model for a mix of spherical grains and air (Guillemoteau etal 2011)"
    por = 1 - bulkdens/partdens
    const = por*(air_perm/bulk_perm)**(-1/3)  # Just for simplify writing
    return (-const*air_perm + bulk_perm)/(1 - const)
    
    
##########################################   Imaginary PERMITTIVITY     ###########################################
    
    
def ipeplinski(vol_moist_cont, bulkdens, partdens, clay_cont, sand_cont, frec, wp, alpha = 0.65 , ewinf = 4.9, tau = 0.58e-10):
    """
    Peplinski 1995 model for imaginary component of permittivity eqs 1 - 6 
    """
    beta2 = 1.33797 - 0.603*sand_cont/100 - 0.166*clay_cont/100
    cond = 0.0467 + 0.2204*bulkdens - 0.4111*sand_cont/100 + 0.6614*clay_cont/100
    efw2 = ((tau*frec*(wp - ewinf))/(1+(tau*frec)**2)) + cond*(partdens - bulkdens)/(2*pi*e_0*frec*partdens*vol_moist_cont)
    iperm = ((vol_moist_cont**beta2)*(efw2**alpha))**(1/alpha)
    return iperm*e_0 


def ipeplogsdon(vol_moist_cont, bulkdens, partdens, clay_cont, sand_cont, frec, boundwat, wp, alpha = 0.65, ewinf = 4.9, tau = 0.58e-10):
    """
    Logsdon 2010 eq for relaxation component (measured with 50*10**6 Hz frec) and Peplinski 1995 for conductivity component of imaginary permittivity    
    """
    beta2 = 1.33797 - 0.603*sand_cont/100 - 0.166*clay_cont/100
    cond = 0.0467 + 0.2204*bulkdens - 0.4111*sand_cont/100 + 0.6614*clay_cont/100
    perm_eff = -6.55 + 22.3*(vol_moist_cont-boundwat) + 161*boundwat
    efw2 = perm_eff + (cond*(partdens - bulkdens)*(vol_moist_cont**beta2/alpha))/(2*pi*e_0*frec*partdens*vol_moist_cont)
    return efw2*e_0 


def sab_ech2(vmc):
    """ Sabouroux & Ba, Progress In Electromagnetics Research B, Vol. 29, 191{207, 2011. ECH2 sample (pure sand). Pag 7, eq 9."""
    return 0.06 + 1.35*vmc - 0.53*vmc**2 + 18.58*vmc**3


def sab_ech3(vmc):
    """ Sabouroux & Ba, Progress In Electromagnetics Research B, Vol. 29, 191{207, 2011. ECH3 sample (sand with 10% clay). Pag 7, eq 10."""
    return 0.03 + 4.1*vmc - 11.3*vmc**2 + 65.35*vmc**3
    
    
def sab_ech4(vmc):
    """ Sabouroux & Ba, Progress In Electromagnetics Research B, Vol. 29, 191{207, 2011. ECH4 sample (sand with 20% clay). Pag 7, eq 11."""
    return 0.14 + 4.63*vmc - 28.04*vmc**2 + 151.67*vmc**3
    
    
def idebye(frec, time, eps_0, eps_inf):
    """Debye model for frequency dependence of polarization process as defined in Glover 2015 pag 27."""
    return (((eps_0-eps_inf)*2*np.pi*frec*time)/(1+((2*np.pi*frec)**2)*time**2))*e_0


def ieq67(frec, cond, iperm):
    """ Imaginary part of total permittivity as defined in eq 68 Glover etal 2015"""
    return (iperm + cond/(2*np.pi*frec))*e_0


################################################### MODIFIED PERMITTIVITY MODELS ###################################


def modpride94(vol_moist_cont, bulkdens, partdens, clay_cont, perm_solid, wp, offset):
    """
    MODIFICATION OF Pride 1994 Equation for permittivity
    """
    por = 1 - bulkdens/partdens
    alpha = (-0.46 * clay_cont/100) + 0.71
    offset = 5
    #m = np.log(((1 - por)*(perm_solid/wp)**alpha) + por) / (np.log(por)*alpha)
    m = (np.log((((1 - por)*(perm_solid/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    #m = 1/alpha
    S = vol_moist_cont / por
    
    return ((por**m) * (wp - perm_solid) + perm_solid)*e_0


def modroth1990(vmc, bulkdens, pd, clay_cont, air_perm, perm_solid, wp): 
    """
    MODIFICATION OF Roth et.al (1990) Equation using Wunderlich et al. empirical relationship among roth alpha exponent and clay content. Also particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below)  
    """
    por = 1 - bulkdens/pd
    alpha = -0.46*clay_cont/100 + 0.71                        
    return (( vmc*wp**alpha + (1-por)*perm_solid**alpha + (por-vmc)*air_perm**(alpha))**(1/alpha))*e_0


def wund13permmod(vol_moist_cont, bulkdens, clay_cont, org_matter, perm_solid, wp, densorg = 1.3, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209, vol_wat_cont_init = 0.04, perm_init=6): 
    """
    #Wunderlich et.al 2013 Equation for permitivity
    Also particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below). This function has the issue of calculing L based on 'm' but that L is for particles, not for pores as is defined in Wunderlich model, reason why this is not theoretically viable. 
    """
    clay = clay_cont/100
    org = org_matter/100
    somr = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
    
    por = 1 - bulkdens/pd
    alpha = (-0.46 * clay) + 0.71
    offset = 5
    #m = np.log(((1 - por)*(perm_solid/wp)**alpha) + por) / (np.log(por)*alpha)
    m = (np.log((((1 - por)*(perm_solid/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    print()
    #m = 1/alpha
    L = (-1/m) + 1
        
    diff = vol_moist_cont - vol_wat_cont_init                      # Diference utilized just for simplicity
                                                                   #Initializing diferenciation parameters
    y = perm_init                                                  # Initial permitivity = Epsilon sub 1  
    x = 0.001                                                      # Diferentiation from p = 0  
    dx = 0.001                                                     # Diferentiation step
                                                                   # Diferentiation until p = 1
    while x<1:                                                    
        dy = ((y*diff)/(1-diff+x*diff)) * ((wp-y)/(L*wp+(1-L)*y))
        x=x+dx
        y=y+dy*dx
    return y*e_0


def modlindeperm(vol_moist_cont, bulkdens, pd, clay_cont, air_perm, perm_solid, wp, n, offset):
    """
    Linde permittivity 2006 modification introducing pro3 equation
    Also particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below)
    """    
    por = 1 - bulkdens/pd
    alpha = -0.46 * clay_cont/100 + 0.71
    m = (np.log((((1 - por)*(perm_solid/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    #print("m Linde perm", m)
    n = m
    #m = 1/alpha
    S = vol_moist_cont / por
    
    return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*perm_solid + (1 - S**n)*air_perm))*e_0


def mod2lindeperm(vol_moist_cont, bulkdens, pd, clay_cont, air_perm, perm_solid, wp):
    """
    Linde permittivity 2006 modification introducing pro3 equation
    Also particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below)
    """    
    por = 1 - bulkdens/pd
    
    if [clay_cont >= 5]:
        m = 0.92*clay_cont**0.2
    else:
        m = 1.25

    n = m
    #m = 1/alpha
    S = vol_moist_cont / por
    
    return ((por**m) * ((S**n)*wp + ((por**-m) - 1)*perm_solid + (1 - S**n)*air_perm))*e_0


def modsen1981(vol_moist_cont, bulkdens, pd, clay_cont, air_perm, perm_solid, wp, offset):        
    """
    # Sen et.al 1981 Equation modified introducing prop3 link between L and Clya_cont, por, perm_solid and wp
    Also particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below)
    """
    por = 1 - bulkdens/pd
    
    alpha = -0.46 * clay_cont/100 + 0.71
    #m = np.log(((1 - por)*(perm_solid/wp)**alpha) + por) / (np.log(por)*alpha)
    m = (np.log((((1 - por)*(perm_solid/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    #m = 1/alpha
    L = (-1/m) + 1
    #print("Mod Sen81 L = ", L)
    cl = vol_moist_cont*(L*perm_solid + (1-L)*wp)                         # Calculation just for simplicity 
    wcg = wp*(((1-por)*perm_solid+cl) / ((1-por)*wp+cl))                  # wcg = wat coated grains
    df = (por*-1) + vol_moist_cont                                        # Diference utilized just for simplicity
                                                                          # Initializing diferenciation parameters
    y = air_perm                                                          # Initial permitivity = Epsilon sub a  
    x = 0.001                                                             # Diferentiation from p = 0  
    dx = 0.01                                                             # Diferentiation step
    
    while x<1:                                                            # Diferentiation until p = 1
        dy = ((y*(1+df))/(-df+x*(1+df))) * ((wcg-y)/(L*wcg+(1-L)*y))
        x=x+dx
        y=y+dy*dx
    return y*e_0 
     
    
def modfengsen1985(vol_moist_cont, bulkdens, pd, clay_cont, air_perm, perm_solid, wp, offset, densorg = 1.3, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """
    # Feng & Sen 1985 Equation modified introducing prop3 link between L and Clya_cont, por, perm_solid and wp
    Also particle density as function of organic matter and clay content as is described in Schjonning et al 2012 (def schjonnpd below)
    """
    por = 1 - bulkdens/pd
    alpha = -0.46 * clay_cont/100 + 0.71
    #m = np.log(((1 - por)*(perm_solid/wp)**alpha) + por) / (np.log(por)*alpha)
    m = (np.log((((1 - por)*(perm_solid/wp)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    #m = 1/alpha
    L = (-1/m) + 1
    
    vmc = vol_moist_cont                                          # Abreviation just for simplicity 
    pds = perm_solid                                              # Abreviation just for simplicity
                                                                  # Initializing diferenciation parameters
    y = wp                                                # Initial permitivity = Epsilon sub a  
    x = 0                                                         # Diferentiation from p = 0  
    dx = 0.001                                                    # Diferentiation step
    
    while x<1:                                                    # Diferentiation until p = 1
        dy = (y/(vmc+x*(1-vmc))) * ((((1-por)*((pds-y))/(L*pds+(1-L)*y))) + ((por-vmc)*(air_perm-y))/(L*air_perm+(1-L)*y)) 
        x = x + dx
        y = y + dy*dx
    return y*e_0  


########################################## BULK DENSITY PEDOTRANSFER FUNCTIONS #######################################


def adams(loi):
    return  100/((loi/0.311) + ((100-loi)/1.47)) 


def Adamsrec(LOI):
    """
    """
    return 100/((LOI/0.312) + ((100 - LOI)/1.661))


def alexander(oc):
    """
    Alexander (1980) R2 = 0.46, n = 721
    """
    return 1.660 - 0.308*oc**0.5


def bernoux(sand, oc):
    """
    (Bernoux et al., 1998)
    """
    return 0.0181*sand - 0.08*oc

    
def botula1(sand, clay, org_matter):
    """
    Botula et al. 2015 using samples from Congo
    """
    return 1.64580 - 0.00362*clay - 0.0016*sand - 0.0158*org_matter


def botula2(org_matter, Fe, Al):
    """
    Botula et al. 2015 using samples from Congo
    """
    return 1.54373 - 0.02287*org_matter + 0.01518*Fe - 0.15198*Al


def de_vos(loi):
    """
    Best model on De Vos et al. 2005 for soils in Belgium eq7
    """
    return 1.775 - 0.173*loi**0.5

def de_vos_ba(loi, depth):
    """
    Ba model on De Vos et al. 2005 for soils in Belgium
    """
    return 1.6417 - 0.1526*loi**0.5 + 0.0144*depth**0.5

def de_vos_co(loi):
    """
    Co model on De Vos et al. 2005 for soils in Belgium
    """
    return (0.6021 - 0.0260*loi)**-1

def de_vos_ca(clay, loi, depth):
    """
    Ca model on De Vos et al. 2005 for soils in Belgium
    """
    return (0.6035 + 0.0233*loi + 0.0034*clay - 0.0004*depth)**-1


def Federer(LOI):
    """
    """
    orm = LOI/100
    return np.exp( -2.31 - 1.079*np.log(orm) - 0.113*np.log(orm)**2 )


def HarrisonandBocockt(LOI):
    """
    (1981) R2 = 0.81, n = 539
    """
    return 1.558 - 0.728*np.log10(LOI) 

def HarrisonandBococks(LOI):
    """
    (1981) R2 = 0.58, n = 538
    """
    return 1.729 - 0.769*np.log10(LOI)


def hollis(sand, clay, org_matter):
    """
    Hollis et al. (2012) European soils
    """
    return 0.80806 + 0.823844*np.exp(-0.27993*org_matter) + 0.0014065*sand - 0.0010299*clay


def HoneysettandRatkowsky(LOI):
    """
    R2 =  0.96, n = 136
    """
    return (0.548 + 0.0588*LOI)**-1


def Huntington(LOI):
    """
    R2 = 0.75, n = 60
    """
    orm = LOI/1000
    return np.exp( -2.39 - 1.316*np.log(orm) - 0.167*np.log(orm)**2 ) 


def jeffrey(LOI):
    """
    Jeffrey 1970 r2 = 0.82, n =80
    """
    return 1.482 - 0.6786*np.log10(LOI)


def Kaur(silt, clay, oc):
    """
    Kaur et al 2002. R2 = 0.62, n = 224 
    """
    return np.exp ( 0.313 - 0.191*oc + 0.02102*clay - 0.000476*clay**2 - 0.00432*silt )


def leonaviciuteA(silt, clay, oc):
    """
    A horizont, n = 993
    """
    return 1.70398 - 0.00313*silt + 0.00261*clay - 0.11245*oc
    
def leonaviciuteE(sand, silt, clay, oc):
    """
    E horizont, n = 993
    """
    return 0.99915 - 0.00592*np.log(silt) + 0.07712*np.log(clay) + 0.09371*np.log(sand) - 0.08415*np.log(oc)

def leonaviciuteB(sand, silt, clay, oc):
    """
    B horizont, n= 993
    """
    return  1.07256 + 0.032732*np.log(silt) + 0.038753*np.log(clay) + 0.078886*np.log(sand) - 0.054309*np.log(oc)

def leonaviciuteBCC(sand, silt, clay, oc):
    """
    Bc-C horizont, n=993
    """
    return 1.06727 - 0.01074*np.log(silt) + 0.08068*np.log(clay) + 0.08759*np.log(sand) + 0.05647*np.log(oc)


def ManriqueandJones(oc):
    """
    Manrique and jones 1991. R2 = 0.41, n = 19651
    """
    return 1.660 - 0.318*oc*0.5


def RawlsandBrakensiek(partdens, LOI):
    """
    """
    return 100/((LOI/0.224) + ((100 - LOI)/partdens))


def TamminenandStarr(LOI):
    """
    R2 = 0.61, n = 158
    """
    return 1.565 - 0.2298*(LOI)**0.5


def modroth_por(vmc, clay, perm_bulk, air_perm, perm_solid, wp):
    """
    Modification of Roth 1990 volumetric mixing model for porosity"""
    alpha = -0.46*clay/100 + 0.71  
    num = (perm_bulk**alpha) - (vmc*wp**alpha) - (perm_solid**alpha) + (vmc*air_perm**alpha)
    den = (air_perm**alpha) - (perm_solid**alpha)
    return num/den


def LRMod1_por(vmc, clay_cont, bulk_cond, cond_matrix, rest_wat):    
    """Lichtenecker and Rother (1931), Korvin (1982) Derived from the theory of functional equations under appropriate
boundary conditions. Formally, the same as Archie’s law if s1=0. Glover 2015 Table 2. 'm' is calculated using prop9.
       Adapted for porosity"""
    alpha = -0.46 * clay_cont/100 + 0.71
    num = -bulk_cond**alpha + vmc*(1/rest_wat)**alpha 
    den = cond_matrix**alpha
    return (num/den) + 1

#### Tranter 2007 ρm = a + b × sand % + (c − sand %)2 × d + e × log depth
#### buscar jalabert et al 2010


################################# # # # # #    PARTICLE DENSITY    # # # # ######################################


def schjonnpd(clay, org, densorg = 1.4, denspart = 2.65, densclay = 2.86, a = 1.127, b = 0.373, c = 2.648, d = 0.209):
    """
    Corrected emprical equation in Schjonnen et al. 2017 for Danish soils and tested on a total of 227 soils from Hansen (1976), Keller and Håkansson (2010)(Scandinavia) and Joosse and McBride(2003) (North America). The soils represent topsoil horizons rich in SOM, as well as samples from B and C horizons"""
    clay = clay/100
    org = org/100
    somr = (org*densorg)/(org*densorg + (1-org)*denspart)
    claymass = (clay*densclay)/(clay*densclay + (1-clay)*denspart)
    pd = ((somr/(a+b*somr)) + (1-somr)/(c+d*claymass))**-1
    return pd


def jacob89_pd(clay, org):
    """Schjonnen et al. 2017 using data of Jacobsen 1989. Clay and ORG are expressed in mass fraction"""
    return 2.652 + 0.216 * clay - 2.237 * org

    
def ruhlmann(org, densorg = 1.4, denspart = 2.65, a = 1.127, b = 0.373):
    """
    Rühlmann et al. (2006)
    """
    org = org/100
    somr = (org*densorg)/(org*densorg + (1-org)*denspart)
    pd = ((somr/(a+b*somr)) + (1-somr)/(2.684))**-1
    return pd


def mcbridepd(org):
    """
    McBride et al 2012
    """
    pd = 2.714 - (0.0198*org)
    return pd


##################################     CLAY CONTENT STIMATIONS     ##########################################


def jacandschj93B_clay(vol_moist_cont, bulkdens, org, aperm):
    """
    #Jacobsen and Schjonning (1993) Equation (2) for clay content. Here the permittivity is the aparent one.
    """
    return (vol_moist_cont + 0.0341 - (0.0345)*aperm + (0.00114)*(aperm**2) - (0.0000171)*(aperm**3) + (0.037)*(bulkdens) - (0.00477)*(org))/(0.000736)


##################################     ORGANIC MATTER STIMATIONS     ##########################################


def devos_inv(bulkdens):
    """
    Best model on De Vos et al. 2005 for soils in Belgium
    """
    return ((-bulkdens+1.775)/0.175)**2


def Banin_Amiel_org(clay, cec):
    """
    CEC [meq/100g] vs clay and organic matter [%], as empirically derived in Banin & amiel 1969 """
    return (-cec + 0.703*clay + 5.1)/2.98


def fernandez88wet(nusell):
    """"Fernandez et al 1988, "Color, Organic Matter, and Pesticide Adsorption Relationships in a Soil Landscape".
    PTF relatinf Nusell values with organic matter [g/kg] for dry and wet samples taken from Indiana, USA. All samples selected had from 58 to 75% silt and from 12 to 35% clay.
    n = 12, r2 = 0.94"""
    return (-nusell+4.38)/0.0523


def fernandez88dry(nusell):
    """"Fernandez et al 1988, "Color, Organic Matter, and Pesticide Adsorption Relationships in a Soil Landscape".
    PTF relatinf Nusell values with organic matter [g/kg] for dry and wet samples taken from Indiana, USA. All samples selected had from 58 to 75% silt and from 12 to 35% clay.
    n = 12, r2 = 0.92"""
    return (-nusell+6.33)/0.0511


######################################## Specific Surface Area (SSA) Pedotransfer ##############################


def Banin_Amiel1(clay):
    """
    Specific Surface Area [m2/g] vs clay content, as empirically derived in Banin & amiel 1969 """
    return 5.76*clay-15.064

""" It is mentioned in Revil etal. 1998 Fgirue 1 an empirical linear relationship among SSA and CEC for rocks"""


def Banin_Amiel2(cec):
    """
    Specific Surface Area [m2/g] vs cec, as empirically derived in Banin & amiel 1969 """
    return (cec-3.23)/0.119


def wang(bound_wat):
    """
    Specific Surface Area [m2/g] vs bound water [% by weight] (called tightly bound water content in this paper) for soils with organic matter. Wang et al. 2011. R2 = 0.827 """
    return 25.2*bound_wat   

    
######################################## Cation Exchamge Capacity (CEC) Pedotransfer ##############################


def Banin_Amiel3(clay, org):
    """
    CEC [meq/100g] vs clay and organic matter, as empirically derived in Banin & amiel 1969 """
    return 0.703*clay - 2.98*org + 5.1


def shah_singh1(clay):
    """ Shah & Singh empirical model. CEC [meq/100g]"""
    cec = 4.18 + 0.62*clay                            
    return cec
    
    
def bell_keulen95A(clay, org, ph):
    """Bell & Keulen 1995, Soil Pedotransfer Functions for Four Mexican Soils. n = 148 using clay loam (alfisol), sandy loam (entysol), clay (vertisol) and silt loam (alfisol) in Mexico. r2 = 0.94. CEC [cmolc/kg] """
    cec = -10 + 0.163*org*ph - 0.0209*org*clay + 0.131*clay*ph
    return cec
    
    
def bell_keulen95B(clay, org, ph):
    """Bell & Keulen 1995, Soil Pedotransfer Functions for Four Mexican Soils. n = 148 using clay loam (alfisol), sandy loam (entysol), clay (vertisol) and silt loam (alfisol) in Mexico. r2 = 0.96. CEC [cmolc/kg] """
    cec = 42.8 - 5.36*ph +0.297*org - 2.04*clay + 0.363*clay*ph 
    return cec


def mcbratney02(clay, org):
    """McBratney et al. 2002, Eq 10. Empirically developed using a soil database of n=1930 (r2=0.739)aparently exposed in McGarry et al 1989 (McGarry, D., Ward, W.T., McBratney, A.B., 1989. Soil Studies in the Lower Namoi Valley: Methods and Data. The Edgeroi Data Set. CSIRO Division of Soils, Glen Osmond, South Australia.) I did not have accse to. cec[mmol+/kg]. org is actually oragnic carbon."""
    cec = -29.250 + 8.139*clay + 0.253*clay*org
    return cec
    
    
######################################     Cementation exponent (m) Pedotransfer     ########################


def mendelson82(m):
    """Mendelson & Cohen 1982 discrete formula (Eq 28) relating L and m for saturated non-clay rocks 
    with randomly oriented elipsoidal grains for which La + Lb + Lc = 1"""
    l2p = []
    for i in range(len(m)):
        l2 = [-3*m[i], 3, -5+3*m[i]]
        roots = np.roots(l2)
        roots = roots[roots.imag == 0 ]
        l2 = roots[roots > 0]
        if l2.size==0:
            l2p.append(np.nan)
        else:    
            l2p.append(l2[0].real)
    return  l2p


def Grunzel(cec):
    """Grunzel 1994 as described in Shah & Singh 2005."""
    return 1.67 + 0.1953*cec**0.5


def prop15(clay_cont):
    """Shah & Singh 2005 eq 12b and 12d describing 'm' in funtion of clay content"""
    
    if clay_cont >= 5:
        m = 0.92*clay_cont**0.2
    else:
        m = 1.25
    return m


def schwartza(clay_cont):
    """  F. Schwartza,b,*, Mazdeline E. Schreibera, Tingting Yan. 2008  """
    return 0.485*(clay_cont)**0.0818


################################# # # # # QP TO APARENT CONDUCTIVITY # # # # ######################################


def mcneill_eca(qp, frequency, coil_spacing):
    """
    Calculate the apparent electrical conductivity (ECa) based on the raw sensor output (Hs/Hp in ppt).
    >> insert the relevant equation between the square brackets below, 
    >> assigning the output to the variable eca (eca = []) 
    """
    # Constants 
    PI = np.pi
    MU_0 = 4 * PI * 10 ** (-7)
    
    # Angular frequency
    angular_frequency = 2 * PI * frequency

    # Equation to transform the appropriate signal response to ECa, using the low-induction number approximation
    # eca = [] #
    eca = (4 * qp) / (angular_frequency * MU_0 * coil_spacing ** 2)
    return eca


################################### Susceptibility ############################################


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


#################################### # # # OUR OWN PTFs PROPOSES  # # # ########################################


def prop1(L):
    a = -1/0.46
    b = 0.71/0.46
    m = ((5/3) - L)/(1 - L**2)
    cc = a/m + b
    return (16.2 + 1.314*cc*100)/1000


def prop2(L):
    a = -1/0.46
    return (((1 - L**2)/(5/3 - L)) - 0.71)*100*a


def prop3(bd, pd, clay_cont, wp, perm_solid = 5, partdens=2.65):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (saturated condition for permittivity mixing model), 
    8, 9 and Archi's law
    And finally we return the value of L
    """
    por = 1- bd/pd
    alpha = -0.46 * clay_cont/100 + 0.71
    m = np.log(((1 - por)*(perm_solid/wp)**alpha) + por) / (np.log(por)*alpha)
    return ((-1/m) + 1)


#def prop5(vol_moist_cont, clay_cont, organic_cont, _bulk_drydens, rest_wat, wp):
    
#    p = [17.1e-6*(wp**3)*(rest_wat**3), -11.4e-4*(wp**2)*(rest_wat**2), 3.45e-2*(wp)*(rest_wat), 
#         -3.41e-2 - vol_moist_cont -3.7e-2 * _bulk_drydens + 7.36e-4 * clay_cont + 47.7e-4 * organic_cont]
#    roots = np.roots(p)
#    roots = roots[roots.imag == 0 ]
#    cond = roots[roots > 0]
#    return cond[0].real


def prop6(vol_moist_cont, por, clay_cont, air_perm, perm_solid, wp):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 4 (for unsaturated consitions), 8, 9 and Archie law for 
    saturated conditions
    And finally we return the value of L
    """
    
    alpha = -0.46 * clay_cont/100 + 0.71
    m = np.log(((1 - por)*(perm_solid/wp)**alpha) + vol_moist_cont + (por-vol_moist_cont)*(air_perm/wp)**alpha) / (np.log(por)*alpha)
    return ((-1/m) + 1)


def prop7(vol_moist_cont, por, clay_cont, air_perm, perm_solid, n, wp):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 4 (for unsaturated consitions), 8, 9
    and Archie law for unsaturated conditions.
    Finally we return the value of L
    """
    S     = vol_moist_cont/(por)
    alpha = -0.46 * clay_cont/100 + 0.71
    m     = np.log(((1 - por)*(perm_solid/wp)**alpha) + vol_moist_cont + (por-vol_moist_cont)*((air_perm/wp)**alpha) - alpha*n*np.log(S))/ (np.log(por)*alpha)
    L     = (-1/m) + 1
    return L
 

def prop8(por, clay_cont, wp, perm_solid = 5):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (saturated condition for permittivity mixing model), 
    8, 9 and Archi's law
    And finally we return the value of L using Han et l. 2020 eq 5
    """
    alpha = -0.46 * clay_cont/100 + 0.71
    m = np.log(((1 - por)*(perm_solid/wp)**alpha) + por) / (np.log(por)*alpha)
    p = [-3*m, 3, 3*m-5]
    roots = np.roots(p)
    roots = roots[roots.imag == 0 ]
    cond  = roots[roots > 0]
    
    return cond


def prop9(clay_cont):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (saturated condition for permittivity mixing model), 
    8, 9 and Archi's law
    And finally we return the value of m
    """
    alpha = -0.46 * clay_cont/100 + 0.71
    m     = 1/alpha
    return m


def prop10(bulkdens, partdens, clay_cont, wp, offset):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (saturated condition for permittivity mixing model), 
    8, 9 and Archi's law
    And finally we return the value of m
    """
    rate = 0
    alpha  = -0.46 * clay_cont/100 + 0.71
    por    = 1 - bulkdens/partdens              
    m      = (np.log((((1 - por)*(rate)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    return m


def prop11(bulkdens, partdens, clay_cont, perm_sol, wp):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (saturated condition for permittivity mixing model), 
    8, 9 and Archi's law
    And finally we return the value of m
    """
    rate  = perm_sol/wp
    alpha = -0.46 * clay_cont/100 + 0.71
    por   = 1 - bulkdens/partdens              
    m     = np.log(((1 - por)*(rate)**alpha) + por) / (np.log(por)*alpha)
    return m


def prop12(bulkdens, partdens, clay_cont, perm_sol, wp, offset):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (saturated condition for permittivity mixing model), Hilhorst 2000 equation and Archi's law
    And finally we return the value of m
    """
    rate   = perm_sol/wp
    alpha  = -0.46 * clay_cont/100 + 0.71
    por    = 1 - bulkdens/partdens                    # deducted from definition of por without assumptions in a three phase medium
    m      = (np.log((((1 - por)*(rate)**alpha) + por)**(1/alpha) - (offset/wp))) / np.log(por)
    return m


def prop13(vmc, bulkdens, partdens, clay_cont, perm_sol, wp, offset, n):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (saturated condition for permittivity mixing model), Hilhorst 2000 equation and Archi's law
    And finally we return the value of m
    """
    alpha  = -0.46 * clay_cont/100 + 0.71
    por    = 1 - bulkdens/partdens  
    S      = vmc/por
    rate   = perm_sol/wp
    m        = np.log(((((1-por)*rate**alpha) + vmc + ((por-vmc)*(1/wp)**alpha))**(1/alpha)- (offset/wp))*S**(-n))/ np.log(por)
    return m


def prop14(vmc, bulkdens, partdens, clay_cont, perm_sol, wp, m, offset):
    """
    First equation (empirical) relating alpha with clay content is taken from Wunderlich et al. 2013
    The second one is deducted from Brovelli & Casianni 2008 convining eqs. 5 (non-saturated condition for permittivity mixing model), Hilhorst 2000 equation and combined Archi's law. And finally we return the value of n"""
    
    alpha  = -0.46 * clay_cont/100 + 0.71
    por    = 1 - bulkdens/partdens  
    S      = vmc/por
    numerator = ((1-por)*perm_sol**alpha + vmc*wp**alpha + (por-vmc))**(1/alpha) 
    return np.log((numerator - offset)/((por**m)*wp))/np.log(S)

    
def prop16(clay_cont):
    """
    Combination of Grunzet thesis 1994 pedotransfer function (I could not access to) and Shah & Singh 2005 """
    cec = 4.18 + 0.62*clay_cont                # Fig 5, Shah & Singh 2005 [meq/100g]
    m   = 1.6 + 0.1953*cec**0.5                # Grunzel 1994 as described in Shah & Singh 2005. Original expression: 1.6+0.1953*cec**0.5  
    return m

############## Utils 
def RMSE(predictions, targets):
    """
    Compute the Root Mean Square Error.

    Parameters:
    - predictions: array-like, predicted values
    - targets: array-like, true values

    Returns:
    - RMSE value
    """
    differences = np.array(predictions) - np.array(targets)
    return np.sqrt(np.mean(differences**2))