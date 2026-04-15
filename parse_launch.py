import yaml
import os
import launch_ros.utilities

pkg_path = '/home/manoja/msp2/upper_body'
ompl_file = os.path.join(pkg_path, 'moveit', 'ompl_planning.yaml')
with open(ompl_file, 'r') as f:
    ompl_config = yaml.safe_load(f)

ompl_config['planning_plugin'] = 'ompl_interface/OMPLPlanner'
ompl_config['request_adapters'] = [
    'default_planner_request_adapters/AddTimeOptimalParameterization'
]

planning_pipeline = {
    'planning_pipelines': ['ompl'],
    'default_planning_pipeline': 'ompl',
    'ompl': ompl_config
}

print("SERIALIZED PARAMS:")
print(yaml.dump(planning_pipeline))
