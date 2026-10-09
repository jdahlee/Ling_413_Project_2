import os

from program_params import CORPUS_DOCUMENTS_FOLDER

def organize_corpus_documents():
    print("Beginning to organize corpus documents by country...")
    #Set root location
    root = os.path.join(".", CORPUS_DOCUMENTS_FOLDER)

    #Look at each file
    count = 0
    for file in os.listdir(root):
        if file.endswith(".txt"):
            #Get the text id
            text_id = file.split(".")[0]
            country = file.split(".")[1]
            
            #Make dir
            os.makedirs(os.path.join(root, country), exist_ok=True)

            os.rename(os.path.join(root, file), os.path.join(root, country, file))
            if (count % 10000 == 0):
                print(f"Organized {count} documents")
            count += 1

if __name__ == "__main__":
    organize_corpus_documents()