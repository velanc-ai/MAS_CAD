import cadquery as cq
import os

# Step 1: Create the base flange cylinder
base_flange = cq.Workplane("XY").circle(124 / 2).extrude(19)

# Step 2: Create the central cylinder
central_cylinder = cq.Workplane("XY").circle(90 / 2).extrude(141)

# Combine the base flange and central cylinder
combined = base_flange.union(central_cylinder)

# Step 3: Create the central hole
result = combined.faces(">Z").workplane().hole(36, 160)

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