import json
import gc
import os
import shutil
import zipfile
from pathlib import Path

import pytest

from autofit.aggregator import Aggregator
from autofit.aggregator.search_output import SearchOutput


@pytest.fixture(name="scan_directory")
def make_scan_directory(tmp_path):
    source = Path(__file__).parent / "search_output"
    destination = tmp_path / "search_output"
    shutil.copytree(source, destination)
    return tmp_path


@pytest.fixture(name="zipped_directory")
def make_zipped_directory(tmp_path):
    source = Path(__file__).parent / "search_output"
    with zipfile.ZipFile(tmp_path / "search_output.zip", "w") as f:
        for path in source.rglob("*"):
            if path.is_file():
                f.write(path, path.relative_to(source))
    return tmp_path


def test_from_directory(scan_directory):
    aggregator = Aggregator.from_directory(scan_directory)
    assert len(aggregator) == 1


def test_search_outputs_are_sorted_by_path(tmp_path, monkeypatch):
    """
    ``os.walk`` yields directory entries in whatever order the filesystem
    reports — hash order on ext4, insertion order on tmpfs — so the aggregated
    outputs must be sorted before they are returned. Without the sort the order
    is machine-dependent, and everything that pairs a row to an output by index
    (the summary CSV writers, ``add_label_column``) pairs them differently on
    different machines.
    """
    source = Path(__file__).parent / "search_output"
    for name in ("a", "b"):
        shutil.copytree(source, tmp_path / name)

    real_walk = os.walk

    def reversed_walk(top, **kwargs):
        for root, dirs, filenames in real_walk(top, **kwargs):
            dirs.reverse()
            yield root, dirs, filenames

    monkeypatch.setattr(os, "walk", reversed_walk)

    aggregator = Aggregator.from_directory(tmp_path)

    assert [output.directory.name for output in aggregator] == ["a", "b"]


def test_zip_extracted_and_loaded(zipped_directory):
    aggregator = Aggregator.from_directory(zipped_directory)
    assert len(aggregator) == 1
    assert (zipped_directory / "search_output" / "files" / "search.json").exists()


def test_zip_not_re_extracted(zipped_directory):
    """
    An extracted directory that carries ``.completed`` is the search, so the zip
    beside it is left alone and the directory on disk is not overwritten.
    """
    Aggregator.from_directory(zipped_directory)

    completed = zipped_directory / "search_output" / ".completed"
    completed.write_text("not the archived content")

    aggregator = Aggregator.from_directory(zipped_directory)

    assert len(aggregator) == 1
    assert completed.read_text() == "not the archived content"


def test_incomplete_sibling_directory_loses_to_the_zip(zipped_directory):
    """
    A directory beside the zip with no ``.completed`` file is not the extraction
    of a finished search — post-completion writers recreate ``<search>/files/``
    after the real directory was zipped and removed. The zip wins, so the search
    is still aggregated under ``completed_only``.
    """
    extracted = zipped_directory / "search_output"
    (extracted / "files").mkdir(parents=True)
    (extracted / "files" / "search.json").write_text("{}")
    (extracted / "files" / "cache_artifact.json").write_text("{}")

    aggregator = Aggregator.from_directory(zipped_directory, completed_only=True)

    assert len(aggregator) == 1
    assert (extracted / ".completed").exists()


def test_zip_temporary_leaves_no_extracted_directory(zipped_directory):
    aggregator = Aggregator.from_directory(zipped_directory, unzip_temporary=True)

    assert len(aggregator) == 1
    assert not (zipped_directory / "search_output").exists()
    assert [path.name for path in zipped_directory.iterdir()] == ["search_output.zip"]


def test_zip_temporary_loads_the_same_outputs(zipped_directory, scan_directory):
    temporary = Aggregator.from_directory(zipped_directory, unzip_temporary=True)
    alongside = Aggregator.from_directory(scan_directory)

    assert len(temporary) == len(alongside)
    assert {output.name for output in temporary[0].jsons} == {
        output.name for output in alongside[0].jsons
    }


def test_zip_temporary_removed_when_released(zipped_directory):
    aggregator = Aggregator.from_directory(zipped_directory, unzip_temporary=True)
    # Only the path is kept: holding the extraction itself would keep it alive.
    path = aggregator[0]._temporary_directory.path

    assert path.exists()

    del aggregator
    gc.collect()

    assert not path.exists()


def test_zip_temporary_outlives_the_aggregator(zipped_directory):
    """
    A search output kept after the aggregator is dropped can still be read.
    """
    aggregator = Aggregator.from_directory(zipped_directory, unzip_temporary=True)
    search_output = aggregator[0]

    del aggregator
    gc.collect()

    assert search_output._temporary_directory.is_alive
    assert {output.name for output in search_output.jsons} == {
        "directory.example",
        "model",
        "samples_info",
        "search",
    }


def test_zip_temporary_removed_on_close(zipped_directory):
    with Aggregator.from_directory(zipped_directory, unzip_temporary=True) as aggregator:
        path = aggregator[0]._temporary_directory.path
        assert path.exists()

    assert not path.exists()


def test_zip_temporary_uses_an_existing_extracted_directory(zipped_directory):
    """
    The completed extraction beside the zip is used as-is; nothing is extracted
    into the temporary directory.
    """
    Aggregator.from_directory(zipped_directory)
    assert (zipped_directory / "search_output" / ".completed").exists()

    aggregator = Aggregator.from_directory(zipped_directory, unzip_temporary=True)

    assert len(aggregator) == 1
    assert aggregator[0].directory == zipped_directory / "search_output"
    assert aggregator[0]._temporary_directory is None


def test_zip_temporary_incomplete_sibling_directory_loses_to_the_zip(zipped_directory):
    """
    The temporary-extraction path has the same precedence: a sibling directory
    without ``.completed`` does not stand for the search, so the zip is extracted
    into the temporary directory and the sibling is left untouched on disk.
    """
    extracted = zipped_directory / "search_output"
    (extracted / "files").mkdir(parents=True)
    (extracted / "files" / "search.json").write_text("{}")

    aggregator = Aggregator.from_directory(
        zipped_directory,
        completed_only=True,
        unzip_temporary=True,
    )

    assert len(aggregator) == 1
    assert aggregator[0]._temporary_directory is not None
    assert aggregator[0].directory != extracted
    assert not (extracted / ".completed").exists()
    assert (extracted / "files" / "search.json").read_text() == "{}"


def test_zip_temporary_mirrors_the_scanned_layout(tmp_path):
    source = Path(__file__).parent / "search_output"
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        with zipfile.ZipFile(directory / "search_output.zip", "w") as f:
            for path in source.rglob("*"):
                if path.is_file():
                    f.write(path, path.relative_to(source))

    aggregator = Aggregator.from_directory(tmp_path, unzip_temporary=True)

    assert len(aggregator) == 2
    assert len({output.directory for output in aggregator}) == 2


@pytest.fixture(name="zipped_grid_search_directory")
def make_zipped_grid_search_directory(tmp_path):
    """
    A grid search, with one child search, archived as a single zip.
    """
    source = Path(__file__).parent / "search_output"
    staging = tmp_path / "staging"
    staging.mkdir()
    shutil.copytree(source, staging / "child")
    (staging / ".is_grid_search").write_text("my_unique_tag")

    with zipfile.ZipFile(tmp_path / "grid.zip", "w") as f:
        for path in staging.rglob("*"):
            if path.is_file():
                f.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    return tmp_path


def test_zip_temporary_grid_search(zipped_grid_search_directory):
    aggregator = Aggregator.from_directory(
        zipped_grid_search_directory,
        unzip_temporary=True,
    )

    assert len(aggregator.grid_search_outputs) == 1

    grid_search = aggregator.grid_searches()[0]

    assert grid_search.unique_tag == "my_unique_tag"
    # The child is paired with its grid search, which only holds if both were
    # extracted into the same mirrored temporary tree.
    assert len(grid_search.children) == 1
    assert aggregator.grid_search_outputs[0]._temporary_directory is not None
    assert [path.name for path in zipped_grid_search_directory.iterdir()] == ["grid.zip"]


def test_outputs_by_suffix(scan_directory):
    search_output = SearchOutput(scan_directory / "search_output")

    assert {output.name for output in search_output.jsons} == {
        "directory.example",
        "model",
        "samples_info",
        "search",
    }
    assert {output.name for output in search_output.pickles} == {"info"}
    assert {output.name for output in search_output.fits} == {"psf"}
    assert search_output.arrays == []


def test_samples_summary_cached():
    directory = (
        Path(__file__).parent / "summary_files" / "aggregate_summary" / "fit_1"
    )
    search_output = SearchOutput(directory)

    assert search_output.samples_summary is search_output.samples_summary


@pytest.fixture(name="test_mode_directory")
def make_test_mode_directory(tmp_path):
    """
    Mirrors the on-disk layout a search produces under test mode: the results
    live beneath an inserted ``test_mode`` segment (``output/test_mode/prefix``)
    while the caller points ``from_directory`` at the real-run location
    (``output/prefix``), which holds no search output of its own.
    """
    source = Path(__file__).parent / "search_output"
    real_directory = tmp_path / "output" / "prefix"
    real_directory.mkdir(parents=True)
    shutil.copytree(source, real_directory.parent / "test_mode" / "prefix" / "search_output")
    return real_directory


def test_from_directory_test_mode_fallback(test_mode_directory, monkeypatch):
    monkeypatch.setattr(
        "autofit.aggregator.aggregator.is_test_mode", lambda: True
    )
    aggregator = Aggregator.from_directory(test_mode_directory)
    assert len(aggregator) == 1


def test_from_directory_no_fallback_when_not_test_mode(test_mode_directory, monkeypatch):
    monkeypatch.setattr(
        "autofit.aggregator.aggregator.is_test_mode", lambda: False
    )
    aggregator = Aggregator.from_directory(test_mode_directory)
    assert len(aggregator) == 0


def _write_search_json(directory: Path, class_path: str):
    files = directory / "files"
    files.mkdir(parents=True)
    (files / "search.json").write_text(
        json.dumps(
            {
                "type": "instance",
                "class_path": class_path,
                "arguments": {},
            }
        )
    )


def test_search_json_is_the_sentinel(tmp_path):
    """
    A directory holding ``files/search.json`` is discovered exactly once, and it
    is the search directory itself which is returned rather than its ``files``
    child.
    """
    _write_search_json(
        tmp_path / "a",
        "autofit.non_linear.search.nest.dynesty.search.static.DynestyStatic",
    )

    aggregator = Aggregator.from_directory(tmp_path)

    assert len(aggregator) == 1
    assert list(aggregator)[0].directory == tmp_path / "a"


def test_files_directory_without_search_json_not_discovered(tmp_path):
    (tmp_path / "b" / "files").mkdir(parents=True)
    (tmp_path / "b" / "files" / "model.json").write_text("{}")

    assert len(Aggregator.from_directory(tmp_path)) == 0


def test_legacy_metadata_still_discovered(tmp_path):
    """
    Output folders written before ``files/search.json`` became the sentinel are
    still aggregated via their ``metadata`` file.
    """
    legacy = tmp_path / "c"
    legacy.mkdir()
    (legacy / "metadata").write_text("name=legacy\nnon_linear_search=emcee\n")

    aggregator = Aggregator.from_directory(tmp_path)

    assert len(aggregator) == 1
    output = list(aggregator)[0]
    assert output.directory == legacy
    assert output.non_linear_search == "emcee"


def test_non_linear_search_from_search_json(directory):
    aggregator = Aggregator.from_directory(directory)

    assert {
        output.directory.name
        for output in aggregator.query(aggregator.non_linear_search == "dynestystatic")
    } == {"search_output", "fit_1", "fit_2"}
    assert [
        output.directory.name
        for output in aggregator.query(aggregator.non_linear_search == "emcee")
    ] == ["search_output_derived"]
