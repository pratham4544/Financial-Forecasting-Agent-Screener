from research.raw import *

DOWNLOAD_DIR = "pdf_downloads"


    
reset_download_folder()

# url = input("Enter Screener URL Link Here : ")
url = 'https://www.screener.in/company/GROWW/consolidated/'

run(url)
print('-'*10)
print('Remove Unwanted PDFs')
print('-'*10)
delete_old_pdfs()
print('Creating Chunks of the PDFs')
chunks = create_chunks(DOWNLOAD_DIR)
print('-'*10)
print('Storing Chunks into Vector DB')
vector_db = create_pdf_vector_stores(chunks)
print('-'*10)
print(f"Added {url} URL as Vector DB")
vector_db = create_url_vector_store(url,vector_db)
print('-'*10)
print('Saving Vector DB into Local')
vector_db.save_local('faiss_index')
print('Loading Local Vector DB')
vector_db = FAISS.load_local('faiss_index',embeddings=embeddings_model,allow_dangerous_deserialization=True)

query = input("Enter Your Query: ")
if query != 'break':
    response = user_query_answer(query,vector_db)
    print(response)
    print('-'*10)
    cmp = current_market_price(url)
    print('-'*10)
    print('Current Market Price of share is :', cmp)
    
