import re
# Read in text data from path
def load_data(path):
    with open(path, 'r') as file:
        text = file.read()
    
    # Split into articles, delimited by  = Valkyria Chronicles III = 
    pattern = r'^\s*=\s[^=].*[^=]\s=$'
    
    # Split the text on matching lines
    sections = re.split(pattern, text, flags=re.MULTILINE)
    

    return sections