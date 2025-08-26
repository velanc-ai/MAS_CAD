import cadquery as cq
import os

# Step 1: Create the base flange
base_flange = cq.Workplane("XY").circle(297 / 2).extrude(13)

# Step 2: Create the cylindrical raised face
raised_face = cq.Workplane("XY").circle(72 / 2).extrude(127)

# Combine the base flange and raised face
combined = base_flange.union(raised_face)

# Step 3: Create the bore
result = combined.faces(">Z").workplane().hole(34)

# Export function for STL
def export_stl(solid, filename=None):
    if filename is None:
        # get the current script name without extension
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        filename = f"{script_name}.stl"
    cq.exporters.export(solid, filename)
    print("Exported to", filename)

# Export the result to STL
export_stl(result)