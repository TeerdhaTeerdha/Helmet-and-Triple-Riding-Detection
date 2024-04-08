from roboflow import Roboflow
rf = Roboflow(api_key="fQ5Ppx2e7ZLYxhdHbJyd")
project = rf.workspace("personal-egpas").project("number-j53ax")
version = project.version(1)
dataset = version.download("yolov8")