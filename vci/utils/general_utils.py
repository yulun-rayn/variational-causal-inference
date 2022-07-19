import json

def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)

def sjson(s, f):
    # create json object from dictionary
    s = json.dumps(s)

    # open file for writing, "w" 
    f = open(f, "w")

    # write json object to file
    f.write(s)

    # close file
    f.close()
