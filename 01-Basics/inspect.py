from metaflow import Flow, get_metadata
print("Current metadata provider: %s" % get_metadata())


run = Flow('PlayListFlow').latest_successful_run
print("Using run: %s" % str(run))