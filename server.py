from flask import Flask, render_template
app = Flask(__name__)

def read_file():
    with open('db.txt', 'r') as f:
        lines = f.readlines()
        nested_lists = list()
        for line in lines:
            values = line.strip().split(',')
            nested_lists.append(values)
    return nested_lists

@app.route('/')
def hello():
    values = read_file()
    return render_template('table.html', title='TESLA STOCK MARKET PREDICTION',users=values)

if __name__ == '__main__':
    app.run(port=7119)