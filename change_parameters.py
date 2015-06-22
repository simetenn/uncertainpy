import string

path = "/home/simen/Dropbox/phd/parameter_estimation/neuron_models/dLGN_modelDB/"
filename = "Parameters.hoc"


def changeParameters(parameters):

    parameterString = """
rall =    $rall 
cap =     $cap
Rm =      $Rm
Vrest =   $Vrest
Epas =    $Epas

gna =     $gna
nash =    $nash
gkdr =    $gkdr
kdrsh =   $kdrsh
gahp =    $gahp
gcat =    $gcat
gcal =    $gcal
ghbar =   $ghbar
catau =   $catau
gcanbar = $gcanbar
    """

    parameterTemplate = string.Template(parameterString)

    filledParameterString = parameterTemplate.substitute(parameters)

    f = open(path + filename, "w")
    f.write(filledParameterString)
    f.close()
    
if __name__ == "__main__":
    parameters = {
        "rall" : 113,   
        "cap" : 1.1,
        "Rm" : 22000,
        "Vrest" : -63,
        "Epas" : -67,
        "gna" :  0.09,
        "nash" : -52.6,
        "gkdr" : 0.37,
        "kdrsh" : -51.2,
        "gahp" : 6.4e-5,
        "gcat" :1.17e-5,
        "gcal" :0.0009,
        "ghbar" :0.00011,
        "catau" : 50,
        "gcanbar" : 2e-8
    }
    changeParameters(parameters)
