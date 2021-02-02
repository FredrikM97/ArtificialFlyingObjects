set -l subcommands add-blob alias-env attach-disk blobs cancel-task clean-up \
    cloud-check cloud-config config configs cpi-config create-env create-release \
    delete-config delete-deployment delete-disk delete-env delete-release \
    delete-snapshot delete-snapshots delete-stemcell delete-vm deploy deployment \
    deployments diff-config disks environment environments errands event events \
    export-release finalize-release generate-job generate-package help ignore \
    init-release inspect-release instances interpolate locks log-in log-out logs \
    manifest orphan-disk recreate releases remove-blob repack-stemcell \
    reset-release restart run-errand runtime-config scp snapshots ssh start \
    stemcells stop sync-blobs take-snapshot task tasks unignore \
    update-cloud-config update-config update-cpi-config update-resurrection \
    update-runtime-config upload-blobs upload-release upload-stemcell variables \
    vendor-package vms

complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'add-blob' --description "Add blob"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'alias-env' --description "Alias environment to save URL and CA certificate"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'attach-disk' --description "Attaches disk to an instance"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'blobs' --description "List blobs"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'cancel-task' --description "Cancel task at its next checkpoint"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'clean-up' --description "Clean up releases, stemcells, disks, etc."
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'cloud-check' --description "Cloud consistency check and interactive repair"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'cloud-config' --description "Show current cloud config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'config' --description "Show current config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'configs' --description "List configs"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'cpi-config' --description "Show current CPI config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'create-env' --description "Create or update BOSH environment"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'create-release' --description "Create release"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-config' --description "Delete config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-deployment' --description "Delete deployment"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-disk' --description "Delete disk"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-env' --description "Delete BOSH environment"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-release' --description "Delete release"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-snapshot' --description "Delete snapshot"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-snapshots' --description "Delete all snapshots in a deployment"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-stemcell' --description "Delete stemcell"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'delete-vm' --description "Delete VM"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'deploy' --description "Update deployment"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'deployment' --description "Show deployment information"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'deployments' --description "List deployments"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'diff-config' --description "Diff two configs by ID"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'disks' --description "List disks"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'environment' --description "Show environment"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'environments' --description "List environments"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'errands' --description "List errands"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'event' --description "Show event details"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'events' --description "List events"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'export-release' --description "Export the compiled release to a tarball"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'finalize-release' --description "Create final release from dev release tarball"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'generate-job' --description "Generate job"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'generate-package' --description "Generate package"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'help' --description "Show this help message"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'ignore' --description "Ignore an instance"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'init-release' --description "Initialize release"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'inspect-release' --description "List release contents such as jobs"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'instances' --description "List all instances in a deployment"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'interpolate' --description "Interpolates variables into a manifest"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'locks' --description "List current locks"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'log-in' --description "Log in"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'log-out' --description "Log out"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'logs' --description "Fetch logs from instance(s)"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'manifest' --description "Show deployment manifest"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'orphan-disk' --description "Orphan disk"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'recreate' --description "Recreate instance(s)"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'releases' --description "List releases"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'remove-blob' --description "Remove blob"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'repack-stemcell' --description "Repack stemcell"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'reset-release' --description "Reset release"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'restart' --description "Restart instance(s)"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'run-errand' --description "Run errand"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'runtime-config' --description "Show current runtime config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'scp' --description "SCP to/from instance(s)"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'snapshots' --description "List snapshots"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'ssh' --description "SSH into instance(s)"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'start' --description "Start instance(s)"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'stemcells' --description "List stemcells"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'stop' --description "Stop instance(s)"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'sync-blobs' --description "Sync blobs"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'take-snapshot' --description "Take snapshot"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'task' --description "Show task status and start tracking its output"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'tasks' --description "List running or recent tasks"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'unignore' --description "Unignore an instance"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'update-cloud-config' --description "Update current cloud config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'update-config' --description "Update config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'update-cpi-config' --description "Update current CPI config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'update-resurrection' --description "Enable/disable resurrection"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'update-runtime-config' --description "Update current runtime config"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'upload-blobs' --description "Upload blobs"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'upload-release' --description "Upload release"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'upload-stemcell' --description "Upload stemcell"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'variables' --description "List variables"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'vendor-package' --description "Vendor package"
complete -f -n "__fish_use_subcommand $subcommands" -c bosh -a 'vms' --description "List all VMs in all deployments"
