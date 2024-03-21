"""
    Functions to work with Excel output
"""

# pylint: disable=C0301,C0103,C0303,C0411

import pandas as pd
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import Alignment

DEFAULT_TEMPLATE = '.template/ExcelTemplate.xlsx'
DOWNLOAD_FILDER  = '.dowbload'

def download_excel(df_excel : pd.DataFrame):
    """Download dataframe as excel"""
    local_excel_buffer = BytesIO()
    df_excel.to_excel(local_excel_buffer, index=False, header=False)
    local_excel_buffer.seek(0)
    return local_excel_buffer


def fill_template(df: pd.DataFrame, template_file_path: str = DEFAULT_TEMPLATE) -> pd.DataFrame:
    """Fill excel template with data"""
    # Load the Excel file
    workbook = load_workbook(template_file_path)

    # Choose the active sheet
    sheet = workbook.active
    
    output_row_index = 3
    
    for _, row in df.iterrows():
        column_name = row['Column']
        column_address = f'{column_name}{output_row_index}'
        sheet[column_address] = row['Answer']
        sheet[column_address].alignment = Alignment(wrap_text=True, vertical='top')
        
    # auto size
    sheet.row_dimensions[output_row_index].height = None
    
    excel_data = BytesIO()
    workbook.save(excel_data)
    excel_data.seek(0)
    
    return excel_data