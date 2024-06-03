from .topfmain import topf
import subprocess
import os
from importlib import resources

if not os.path.isfile(".topf/JuliaEnvironment/Manifest.toml"):
        if __package__ is None:
            raise RuntimeError(
                "The julia setup script for topf must be run as a package."
            )
        try:
            os.mkdir(".topf")
        except FileExistsError:
            pass
        try:
            os.mkdir(".topf/JuliaEnvironment")
        except FileExistsError:
            pass
        manifest_text = resources.read_text(__package__, "JuliaManifest.toml")
        with open(".topf/JuliaEnvironment/Manifest.toml", "w", encoding="utf-8") as f:
            f.write(manifest_text)
        project_text = resources.read_text(__package__, "JuliaProject.toml")
        with open(".topf/JuliaEnvironment/Project.toml", "w", encoding="utf-8") as f:
            f.write(project_text)
        script_text = resources.read_text(__package__, "HomologyGeneratorsMultiD.jl")
        with open(".topf/HomologyGeneratorsMultiD.jl", "w", encoding="utf-8") as f:
            f.write(script_text)
        try:
            process = subprocess.run(
                [
                    "julia",
                    "--project=.topf/JuliaEnvironment",
                    "-e",
                    "using Pkg; Pkg.instantiate()",
                ],
                check=False,
            )
        except Exception as error:
            raise RuntimeError(
                "Could not install Julia dependencies. Please make sure that Julia is installed and the Julia executable is in your PATH."
            ) from error
        if process.returncode != 0:
            raise RuntimeError(
                "Could not install Julia dependencies. Please make sure that Julia is installed and the Julia executable is in your PATH."
            )