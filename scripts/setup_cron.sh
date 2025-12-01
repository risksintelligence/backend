#!/bin/bash
# Setup daily governance report cron job

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_SCRIPT="$BACKEND_DIR/scripts/daily_governance_report.py"

echo "Setting up daily governance report cron job..."
echo "Report script location: $REPORT_SCRIPT"

# Add cron job to run daily at 6:00 AM
(crontab -l 2>/dev/null; echo "0 6 * * * cd $BACKEND_DIR && python $REPORT_SCRIPT >> /var/log/governance_report.log 2>&1") | crontab -

echo "✅ Cron job added: Daily governance report will run at 6:00 AM"
echo "📁 Logs will be written to: /var/log/governance_report.log"

# Test script execution
echo "🧪 Testing script execution..."
cd "$BACKEND_DIR"
python "$REPORT_SCRIPT"

echo "✅ Setup complete!"
