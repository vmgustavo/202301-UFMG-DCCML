import io
import zipfile
import requests

import pandas as pd


def download_data():
    url = 'https://archive.ics.uci.edu/static/public/101/tic+tac+toe+endgame.zip'
    u = requests.get(url)
    f = io.BytesIO()
    f.write(u.content)

    def extract_zip(input_zip):
        input_zip = zipfile.ZipFile(input_zip)
        return {i: input_zip.read(i) for i in input_zip.namelist()}

    extracted = extract_zip(f)

    columns = [
        'top-left-square',
        'top-middle-square',
        'top-right-square',
        'middle-left-square',
        'middle-middle-square',
        'middle-right-square',
        'bottom-left-square',
        'bottom-middle-square',
        'bottom-right-square',
        'Class',
    ]

    data = (
        pd.DataFrame(
            [row.split(',') for row in extracted['tic-tac-toe.data'].decode('utf8').split('\n')],
            columns=columns
        )
        .dropna()
    )

    return data
