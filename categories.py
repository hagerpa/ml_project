import csv
import re
import nltk
from tempfile import TemporaryFile
import numpy as np

class categories:
    """A object of this class keeps track of all the categories in the classification task.
    Its handy since it automatically reads the available categoires and subcategories from
    a CSV file. Its saves mappings beteween categories and subcatgeories.
    More over the class offers a method for printing all categories in a need way, and also
    ofers mehtods for quick acess on category information."""
    
    def __init__(self):
        """ By initializing a categorie object like this, immidiatly avaliable categories are
        read from the category.csv file."""
        self.id_to_dbid = np.array
        self.dbid_to_id = {}
        self.names = np.array
        self.names_to_id = {}
        self.sub_to_id = {}
        self.id_to_subs = {}
        self = self.loadfromfile()
        
    def loadfromfile(self, print_changes=False):
        with open('category.csv', 'r') as file:
            file_content = file.read()
        
        regexps = [(r"\\\"", "'"), 
                   (re.compile(r"<span.*?/span>", re.DOTALL), ""),
                   (r"<p.*?>", ""),
                   (r"</p>", "")]
        
        for find, replace in regexps:
            file_content, n = re.subn(find, replace, file_content)
            if print_changes:
                print("replaced expression {0} by expression {1} - {2} times.".format(find,replace,n))
        
        with TemporaryFile("w+") as file_clean:
            file_clean = TemporaryFile("w+")
            file_clean.write(file_content)
            file_clean.seek(0)
            
            reader = csv.reader(file_clean)
            self.id_to_dbid, self.dbid_to_id, self.names,\
                self.sub_to_id, self.id_to_subs, self.names_to_id = initialize(file_clean, reader)
        
        return self
    
    def __str__(self):
        out = ""
        for i in range(len(self)):
            out += "{0} ({1}): {2} \n".format(i, self.id_to_dbid[i], self[i])
            for subcat in self.id_to_subs[i]:
                out += "\t {0}".format(subcat)
            out += "\n"
        return out
    
    def __len__(self):
        return len(self.id_to_dbid)
    
    def __getitem__(self, ref):
        if type(ref) == int:
            if ref > len(self): raise ValueError("Index out of bounds.")
            return self.names[ref]
        elif type(ref) == str:
            if len(ref) <= 2:
                if int(ref) in self.sub_to_id:
                    return self.sub_to_id[ int(ref) ]
            if ref in self.names_to_id:
                return self.names_to_id[ref]
            else:
                raise Exception("no such category.")
        elif type(ref) == slice:
            return self.names[ref]
        else:
            raise ValueError("As reffrence provide either int or str.")
        
        
def initialize(catfile, catreader):
    catfile.seek(0); next(catreader);
    
    dbids, names = [], []
    
    for row in catreader:
        if int(row[1]) == 0:
            dbids += [ int(row[0]) ]
            names += [ make_feature_name(row[2]) ]
    
    id_to_dbid = np.array(dbids)
    dbid_to_id = { id_to_dbid[i]: i for i in range(len(id_to_dbid)) }
    names = np.array(names)
    
    names_to_id = { names[i]: i for i in range(len(names)) }
    
    sub_to_id = {}
    catfile.seek(0); next(catreader);
    for row in catreader:
        if int(row[1]) != 0:
            sub_to_id[ int(row[0]) ] = dbid_to_id[ int(row[1]) ]
    
    id_to_subs = {}
    for sid, i in sub_to_id.items():
        if i in id_to_subs:
            id_to_subs[i] += [sid]
        else:
            id_to_subs[i] = [sid]

    return id_to_dbid, dbid_to_id, names, sub_to_id, id_to_subs, names_to_id

def make_feature_name(name):
    featurename = name
    regexps = [(r" ", "_"),
                (r"ö","oe"),
                (r"ä","ae"),
                (r"ü","ue"),
                (r"&","and")]
    for find, replace in regexps:
        featurename = re.sub(find, replace, featurename)
    return featurename.lower()