import argparse

def getArgs():
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("--disablePlot", help="increase output verbosity", action='store_true')
    args = argParser.parse_args()
    return args