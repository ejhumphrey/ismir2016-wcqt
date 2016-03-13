import os
import pytest

import wcqtlib.data.cqt as CQT

DIRNAME = os.path.dirname(__file__)


def test_cqt_one(workspace):
    input_file = os.path.join(DIRNAME, "sax_cres.mp3")
    output_file = os.path.join(workspace, "foo.npz")
    assert CQT.cqt_one(input_file, output_file)


def test_download_many(workspace):
    input_files = [os.path.join(DIRNAME, fname)
                   for fname in ("sax_cres.mp3", "mandolin_trem.mp3")]
    output_files = [os.path.join(workspace, fname)
                    for fname in ('foo.npz', 'bar.npz')]
    assert CQT.cqt_many(input_files, output_files)
