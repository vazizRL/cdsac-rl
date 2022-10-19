from object2urdf import ObjectUrdfBuilder

object_folder = "C:/Users/XMG/Desktop/Master/Masterarbeit/Custom_Objects/Obstacle_Refined"
stl_object = "C:/Users/XMG/Desktop/Master/Masterarbeit/Custom_Objects/Obstacle_Refined/obstacle_refined.obj"    # Or .obj data
urdf_prototype = "C:/Users/XMG/Desktop/Master/Masterarbeit/Custom_Objects/Obstacle_Refined/prototype_obstacle_refined.urdf"

builder = ObjectUrdfBuilder(object_folder, urdf_prototype=urdf_prototype)

# builder.build_urdf(filename=stl_object,
#                    force_overwrite=True, decompose_concave=True, force_decompose=False, center='mass')

builder.build_urdf(filename=stl_object,
                   force_overwrite=True, decompose_concave=True, force_decompose=False, center='mass')