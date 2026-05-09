module.exports = {
  apps: [
    {
      name: "task-pool-generator",
      cwd: "/home/const/subnet66/tau",
      script: "./start_task_pool_generator.sh",
      interpreter: "bash",
      autorestart: true,
      min_uptime: "30s",
      max_restarts: 20,
      out_file: "/home/const/subnet66/tau/logs/task-pool-generator-out.log",
      error_file: "/home/const/subnet66/tau/logs/task-pool-generator-error.log",
      merge_logs: true
    }
  ]
};
