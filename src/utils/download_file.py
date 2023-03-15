import base64

def download_link(data, file_name, file_type):
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{file_type};base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return 'Download successful!!!'