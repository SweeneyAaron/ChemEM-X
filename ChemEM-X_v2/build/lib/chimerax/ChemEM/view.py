# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

"""The main HTML view of the ChemEM app"""
from Qt.QtCore import QUrl
from jinja2 import Environment, PackageLoader, select_autoescape
from pathlib import Path
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
        """Render the application when state changes."""
        
        base_dir = Path(__file__).resolve().parent
        base_url = QUrl.fromLocalFile(str(base_dir) + os.sep)
        html = self.template.render(tool=self.tool)
        self.html_view.setHtml(html, base_url)
    
    
        