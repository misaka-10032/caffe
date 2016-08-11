#!/usr/bin/env python
# encoding: utf-8

import os
import json
from functools import wraps
from flask import Flask, render_template, send_from_directory

f_rank = 'eval/rank/rankings.txt'
fmt_bg = '/eval/bg-pool/final_{}.png'
fmt_ln = '/eval/ln-pool/final_{}.png'
fmt_post = '/eval/post-pool/hmap_{}.png'
fmt_post_overlay = '/eval/post-pool/hmap+bg_{}.png'
fmt_ln_overlay = '/eval/ln-pool/ln+bg_{}.png'
fmt_diff_overlay = '/eval/diff-pool/diff+bg_{}.png'

app = Flask(__name__, static_url_path='')
app.config.from_object(__name__)

rank_to_idx = {}
idx_to_rank = {}
with open(f_rank) as f:
   for line in f:
       line = line.strip()
       if not line:
           continue
       rank, idx, score = line.split(' ')
       rank, idx, score = int(rank), int(idx), float(score)
       # simply make them str
       rank_to_idx[rank] = idx, score
       idx_to_rank[idx] = rank, score


def json_resp(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        r = json.dumps(f(*args, **kwargs))
        print r
        return r
    return wrapper


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/eval/<type_>/<fname>")
def send_eval(type_, fname):
    return send_from_directory(os.path.join('eval', type_), fname)


@app.route("/static/<fname>")
def send_static(fname):
    return send_from_directory('static', fname)


def _imgsrc(row, col, rank, score, type_, overlay):
    idx = row * 8 + col
    r = {'row': row, 'col': col, 'rank': rank, 'score': score}
    if overlay == 0:
        if type_ == 0:    # prediction
            r['src'] = fmt_post.format(idx)
        elif type_ == 1:  # ground truth
            r['src'] = fmt_ln.format(idx)
        elif type_ == 2:  # background
            r['src'] = fmt_bg.format(idx)
    elif overlay == 1:
        if type_ == 0:    # prediction
            r['src'] = fmt_post_overlay.format(idx)
        elif type_ == 1:  # ground truth
            r['src'] = fmt_ln_overlay.format(idx)
        elif type_ == 2:  # diff
            r['src'] = fmt_diff_overlay.format(idx)
    return r

@app.route("/imgsrc/rank/<int:rank>/<int:type_>/<int:overlay>")
@json_resp
def imgsrc_rank(rank, type_, overlay):
    idx, score = rank_to_idx[rank]
    row, col = idx//8, idx%8
    return _imgsrc(row, col, rank, score, type_, overlay)


@app.route("/imgsrc/loc/<int:row>/<int:col>/<int:type_>/<int:overlay>")
@json_resp
def imgsrc_loc(row, col, type_, overlay):
    idx = row * 8 + col
    rank, score = idx_to_rank[idx]
    return _imgsrc(row, col, rank, score, type_, overlay)


if __name__ == "__main__":
    app.run()

