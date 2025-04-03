# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""The main HTML view of the ChemEM app"""
from Qt.QtCore import QUrl
from jinja2 import Environment, PackageLoader, select_autoescape
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from chimerax.atomic import AtomicStructure


class ChemEMView:
    def __init__(self, tool):
        self.tool = tool
        self.html_view = tool.html_view

        env = Environment(
            loader=PackageLoader("chimerax.ChemEM", '.'),
            autoescape=select_autoescape(),
        )
        self.template = env.get_template("template.html")

    def render(self):
        """Render the application.  Call this, when the application
        state changes."""
        dir_path = os.path.dirname(__file__)

        qurl = QUrl.fromLocalFile(dir_path)
        html = self.template.render(
            tool=self.tool)
        
        self.html_view.setHtml(html, qurl)
        