# -*- coding: utf-8 -*-

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError,BulkWriteError

from ibslib import Structure 


class StructMongo(MongoClient):
    """
    API for using PyMongo with Structure Files. All functions from the
    MongoClient are inherited.
    """
    def __init__(self):
        super().__init__()
        self.db = None
        self.collection = None
        
        
    def set_database(self,db_name):
        """
        Set database in client
        """
        if db_name not in self.list_database_names():
            raise Exception("Database name for set_database must be one of "+
                            "{}. ".format(self.list_database_names()) +
                            "User argument was: {}.".format(db_name))
        self.db = self[db_name]
        return self.db
    
    
    def set_collection(self, collection_name):
        """
        Set collection in database
        """
        if self.db == None:
            raise Exception("Please call set_database before calling "+
                    "set_collection for StructMongo class.")
        
        # Collection must already exist.
        if collection_name not in self.list_collections():
            raise Exception("Collection name for set_collection must be one "+
                        "of {}. ".format(self.list_collections()) +
                        "User argument was: {}".format(collection_name))
            
        self.collection = self.db[collection_name]
        return self.collection
    
    
    def list_collections(self):
        """
        Get all collection names in current database
        """
        if self.db == None:
            raise Exception("Please call set_database before calling "+
                            "list_collections for StructMongo class.")
        return [x["name"] for x in self.db.list_collections()]
    
    
    def make_database(self, db_name):
        """
        Point self.db to new database. Follows MongoDB standard that a database
        is not created until something is added to it. 
        """
        if db_name in self.list_database_names():
            raise Exception("Argument to make_database already exists: {}. "
                            .format(db_name) +
                            "Please use new database name.")
            
        self.db = self[db_name]
    
    
    def make_collection(self, collection_name):
        """
        Points self.collection to a new collection.
        """
        if self.db == None:
            raise Exception("Please call set_database before calling "+
                            "make_colelction for StructMongo class.")
        if collection_name in self.list_collections():
            raise Exception("Argument to make_collection {} already exists. "
                            .format(collection_name)
                            + "Please use new collection name.")
        
        self.collection = self.db[collection_name]
        
    
    def add_struct(self,struct, _id="", overwrite=False):
        """
        Add or update a structure file to the database.
        
        Arguments
        ---------
        struct: Structure
        _id: str
            _id to be added to document for MongoDB. Default behavior is to  
            use the Structure ID. 
        overwrite: bool
            Will overwrite Structure document if it already exists.             
        """
        if self.collection == None:
            raise Exception("Cannot call add_struct until user has called  "+
                        "set_collection.")
        
        struct_doc = struct.document(_id)
        if overwrite == False:
            try: self.collection.insert_one(struct_doc)
            except DuplicateKeyError as Error:
                raise Exception("Structure {} already exists. ".format(struct_doc["_id"]) +
                        "Use overwrite=True if you wish to overwrite this document.")
        else:
            self.collection.replace_one({"_id":struct_doc["_id"]},
                                        struct_doc, upsert=True)
        
    
    def add_struct_dict(self,struct_dict,_id_list=[],overwrite=False):
        """
        Add or update a StructureDict to the database.
        
        Arguments
        ---------
        struct_dict: StructDict
        _id_list: list of str
            List of _id to be used for each document in the struct_dict. 
            Default behavior is the use the Structure ID. Otherwise, the 
            length of _id_list must be equal to the length of the struct_dict. 
        overwrite: bool
            Will overwrite Structure document if it already exists. 
        """
        if self.collection == None:
            raise Exception("Cannot call struct_dict until user has called  "+
                        "set_collection.")
            
        # Check length of _id_list
        if len(_id_list) != 0:
            if len(_id_list) != len(struct_dict):
                raise Exception("Call add_struct_dict with an _id_list " +
                        "the same length as the struct_dict. " +
                        "len(_id_list) = {}; len(struct_dict) = {}"
                        .format(len(_id_list), len(struct_dict)))
                
        # Have to use add_struct one at a time if overwrite is True
        if overwrite == True:
            for i,struct in enumerate(struct_dict.values()):
                if len(_id_list) == 0:
                    self.add_struct(struct,overwrite=True)
                else:
                    self.add_struct(struct,_id=_id_list[i],overwrite=True)
            return
        
        # Batch addition if overwrite is False
        doc_list = []
        if len(_id_list) == 0:
            for struct_id,struct in struct_dict.items():
                struct_doc = struct.document()
                doc_list.append(struct_doc)
        
        # Using custom _id
        if len(_id_list) != 0:
            i = 0
            for struct_id,struct in struct_dict.items():
                struct_doc = struct.document(_id_list[i])
                doc_list.append(struct_doc)
                i += 1
        
        # Insert many for efficiency
        try: self.collection.insert_many(doc_list)
        except BulkWriteError as Error:
            raise Exception("Pymongo BulkWriteError occured. Try setting overwrite=True.")
    
    
    def get_struct_dict(self):
        """
        Return all documents in the collection as a structure dictionary
        """
        if self.db == None or self.set_collection == None:
            raise Exception("Cannot call get_struct_dict until user has " +
                    "called set_database and set_collection")
        # Convert all documents in collection into Structure objects
        self.struct_dict = {}
        cursor = self.collection.find({})
        for document in cursor:
            struct = Structure.from_dict(document)
            self.struct_dict[struct.struct_id] = struct
        return self.struct_dict
            
        
        
if __name__ == "__main__":
    pass

    sm = StructMongo()
    print(sm.database_names)
    db = sm.set_database("test_struct_database")
    print(sm.list_collections())
    sm.set_collection("FUQJIK")
    print(sm.get_struct_dict())
