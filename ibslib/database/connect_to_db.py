
import json
from ibslib.io import read,write
from pymongo import MongoClient

# Read into structure dictionary
struct_dict = read("FUQJIK_test_structs")

# Connect to Mongodb
client = MongoClient('localhost', 27017)
# # Get all database names
# print(client.list_database_names())

# Creating database is as simple as referencing it
test_struct_db = client["test_struct_database"]
## Creating a collection from database which structures can 
## then be added to. But only need to create a collection once.
# collect = test_struct_db.create_collection("FUQJIK")
collection = test_struct_db["FUQJIK"]

## Writing structures to collection
for struct_id,struct in struct_dict.items():
    print(struct_id)
    struct._id = struct_id
    # Need to convert structure to dictionary
    temp = struct.dumps()
    temp = json.loads(temp)
    temp["_id"] = struct_id
    collection.insert_one(temp)
    pass

## Viewing all contents of collection with cursor
cursor = collection.find({})
for document in cursor:
    print(document["_id"])
    ### Remove all documents
    # collection.remove(document["_id"])

## Number of documents in the collection
# print(collection.count())