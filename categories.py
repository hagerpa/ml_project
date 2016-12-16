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
        self.cats = {}
        self.subcats = {}
        self.loadfromfile()
        self.frequencies = nltk.FreqDist()
        self.internal_id_to_name
        self.internal_name_to_id
    
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
            self.cats, self.subcats = makedict(file_clean, reader)
        
        self.internal_id_to_name = np.array(self.all_names())
        self.internal_name_to_id = { self.internal_id_to_name[i]: i for i in range(len(self)) }
    
    def __str__(self):
        # Prints all categories with subcategories intendet after perent category
        out = ""
        for cat_id, cat in self.cats.items():
            out += str(cat_id) + " " + str(cat[0]) + "\n"
            for subcat in cat[1].items():
                out += "\t" + str(subcat[0]) + " " + str(subcat[1]) + "\n"
            out += "\n"
        return out
    
    def __len__(self):
        return len(self.cats)
    
    def name(self, id_):
        """This method returns the feature name of the category for a given id.
        Note that it will always return the name of the parent categoy if a id
        of a subcategory is given. The feature name is a conected lowercase name
        with refrence to the semantic content of the category.
        """
        # returns the category name of the parent for a given category id
        if id_ in self.cats.keys():
            return self.cats[id_][0]
        elif id_ in self.subcats.keys():
            return self.name(self.subcats[id_])
        else:
            raise ValueError("There is no category with the id", id_)
        
    
    def all_names(self):
        return [n[0] for n in self.cats.values()]
    
    def parent_id(self, id_):
        if id_ in self.subcats.keys():
            return self.subcats[id_]
        else:
            raise ValueError("There is no category with the id", id_)
    
    def internal_id(self, name):
        """ This method returns for a given category name, the new id created while
        loading in the categories. Note that this id doesnt correspond to the
        id given in the database. """
        return self.internal_name_to_id[name]
    
    def __getitem__(self, ref):
        if type(ref) == int:
            if ref > len(self): raise ValueError("index out of bounds.")
            return self.internal_id_to_name[ref]
        if type(ref) == str:
            if len(ref) <= 2:
                if int(ref) in self.subcats:
                    return self[self.name(int(ref))]
            if ref in self.internal_name_to_id:
                return self.internal_name_to_id[ref]
            else:
                raise ValueError("no such category.")
        
        
def initialize(catfile, catreader):
    catfile.seek(0)
    next(catreader)
    category_dict = {} # Dictionary which has as key: category_id, ...
    #..- and as values: (category_feature_name, dictionary of subcategories, category_name)

    subcat = {} # this is a mapping from cadigory ids to parent ids, if it category is a parent
    #.. it'll be the identity

    for row in catreader:
        if int(row[1]) == 0:
            
            category_dict[int(row[0])] = [make_feature_name(row[2]), {}, row[2]]
            subcat[int(row[0])] = int(row[0])
        else:
            subcat[int(row[0])] = int(row[1])
    
    catfile.seek(0)
    next(catreader)

    for row in catreader:
        if int(row[1]) != 0:
            category_dict[int(row[1])][1][int(row[0])] = row[2]

    return category_dict, subcat

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