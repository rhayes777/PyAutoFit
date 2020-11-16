# %%
"""
Tutorial 2: Visualization and Configs
=====================================

In chapter 1, we used the `visualize` method of the `Analysis` class to perform on-the-fly visualization of our
model-fit. The template project of this chapter uses this method, however it defines a `Visualizer` in the `phase`
package and `visualizer.py` module to handle visualization. Furthermore, the plotting functions defined in the `plot`
package are designed to plot the individual attributes of classes such as `FitDataset` and `Dataset.

If you checkout the `visualizer.py` module now, you'll see that the images that are output during a model-fit are
depending on configuration files found in the directory `src/config/plots.ini`. Entries from this configuration file
are loaded using our parent project **PyAutoConf** as follows:

 `conf.instance["visualize"]["plots"]["dataset"]["subplot_dataset"]`

In the previous tutorial, you'lll have noted we told **PyAutoFit** where to load these configs from with the following
Python code:

 from autoconf import conf
 conf.instance.push(new_path=path.join(workspace_path, "howtofit", "chapter_phase_api", "src", "config"))

When you or a user uses your model-fitting software, there are lots of settings that one may wish to customize when
performing model-fitting. This use of config files ensures this is possible, but avoids the user needing to know about
these settings the first time they use the project.

We recommend that you use configuration files to control any aspects of your project which fit this remit. That is,
settings someone may want to change, but are not critical to the project the first time they use it. **PyAutoConf**
can easily be used to do this, whereby adding configuration files following the same structure as the `visualize`
configs above will handle this.

For example, if you had a configuration file in the folder:

 `config/folder1/folder2/file.ini`

And the configuration file read as follows:

[section1]
setting1=True

[section2]
setting2=False

You can load the settings from this configuration file as follows:

 conf.instance["folder1"]["folder2"]["file"]["section1"]["setting1"]
 conf.instance["folder1"]["folder2"]["file"]["section2"]["setting2"]
"""
