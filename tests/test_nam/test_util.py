# File: test_install.py
# File Created: Saturday, 22nd April 2023 5:49:01 pm
# Author: Fábio Silva (silva.fabio@gmail.com)

import os
import shutil
import pytest
from tempfile import NamedTemporaryFile
from nam.util import find_files


class TestFindFiles:
    SEARCH_DIRECTORY = "./tests/fixtures"
    TMP_FILES_PREFIX = "tmp_files_"

    @classmethod
    def setup_class(cls):
        os.makedirs(cls.SEARCH_DIRECTORY, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.SEARCH_DIRECTORY)

    def create_tmp_files(self, n_files: int, extension: str, prefix=TMP_FILES_PREFIX, directory=SEARCH_DIRECTORY):
        tmp_files = []
        for i in range(1, n_files + 1):
            tmp_file = NamedTemporaryFile(suffix=f".{extension}", prefix=prefix, dir=directory)
            tmp_files.append(tmp_file)
        tmp_files = sorted(tmp_files, key=lambda file: file.name)
        return tmp_files

    def test_find_files_with_one_file(self):
        with NamedTemporaryFile(suffix=".wav", prefix=self.TMP_FILES_PREFIX, dir=self.SEARCH_DIRECTORY) as tmp_file:
            found_files = find_files(self.SEARCH_DIRECTORY, "wav")
            assert len(found_files) == 1
            assert os.path.basename(found_files[0]) == os.path.basename(tmp_file.name)

    def test_find_files_with_multiple_files(self):
        self.create_tmp_files(3, "png")
        tmp_files = self.create_tmp_files(3, "wav")

        found_files = find_files(self.SEARCH_DIRECTORY, "wav")
        assert len(found_files) == 3
        for found_file, tmp_file in zip(found_files, tmp_files):
            assert os.path.basename(found_file) == os.path.basename(tmp_file.name)

    def test_find_files_with_exclude_files(self):
        self.create_tmp_files(3, "png")
        tmp_files = self.create_tmp_files(3, "wav")
        tmp_files_excluded = self.create_tmp_files(2, "wav")
        exclude_files=f"{os.path.basename(tmp_files_excluded[0].name)},{os.path.basename(tmp_files_excluded[1].name)}"

        found_files = find_files(self.SEARCH_DIRECTORY, "wav", exclude_files=exclude_files)
        assert len(found_files) == 3
        for found_file, tmp_file in zip(found_files, tmp_files):
            assert os.path.basename(found_file) == os.path.basename(tmp_file.name)

    def test_find_files_with_exclude_files_regex(self):
        exclude_prefix = "tmp_excluded"
        self.create_tmp_files(3, "png")
        tmp_files = self.create_tmp_files(3, "wav")
        exclude_files = self.create_tmp_files(10, "wav", prefix=exclude_prefix)
        exclude_files=f"{exclude_prefix}*"

        found_files = find_files(self.SEARCH_DIRECTORY, "wav", exclude_files=exclude_files)
        assert len(found_files) == 3
        for found_file, tmp_file in zip(found_files, tmp_files):
            assert os.path.basename(found_file) == os.path.basename(tmp_file.name)

    def test_find_files_with_include_files(self):
        self.create_tmp_files(3, "wav")
        self.create_tmp_files(3, "png")
        include_file = self.create_tmp_files(1, "wav")[0]

        found_files = find_files(self.SEARCH_DIRECTORY, "wav", include_files=os.path.basename(include_file.name))
        assert len(found_files) == 1
        assert os.path.basename(found_files[0]) == os.path.basename(include_file.name)

    def test_find_files_with_include_files_regex(self):
        included_prefix = "tmp_included"
        tmp_files = self.create_tmp_files(10, "wav", prefix=included_prefix)
        self.create_tmp_files(5, "wav")
        self.create_tmp_files(3, "png")
        include_files = included_prefix

        found_files = find_files(self.SEARCH_DIRECTORY, "wav", include_files=include_files)
        assert len(found_files) == 10
        for found_file, tmp_file in zip(found_files, tmp_files):
            assert os.path.basename(found_file) == os.path.basename(tmp_file.name)

if __name__ == "__main__":
    pytest.main()
