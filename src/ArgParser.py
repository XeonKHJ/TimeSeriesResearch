import argparse

def getArgs():
    argParser = argparse.ArgumentParser()
    
    argParser.add_argument("--disablePlot", help="Disable drawing plot to notebook.", action='store_true')
    argParser.add_argument("--eval", help="Only outputs evaluate result", action='store_true')
    args, opt = argParser.parse_known_args()
    return args