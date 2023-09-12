#!/bin/bash

# Small utility-script for queueing bash jobs.
# It grabs executes and clears `./scripts/next.sh`.
# Usage: Execute from the project root path, i.e. `./scripts/run_scheduler.sh`

# Load .env
set -o allexport; source ./scripts/queue/.env; set +o allexport

NEXT_SCRIPT=./scripts/queue/next.sh
NEXT_CONTENTS=$(<$NEXT_SCRIPT)
EMPTY=""


while [[ $NEXT_CONTENTS != $EMPTY  ]]
do
    TIMESTAMP=`date "+%Y%m%d%H%M%S"`
    RUN_SCRIPT=./scripts/queue/history/$TIMESTAMP.sh
    mv $NEXT_SCRIPT $RUN_SCRIPT
    echo "" > $NEXT_SCRIPT 
    chmod +x $RUN_SCRIPT
    $RUN_SCRIPT
    NEXT_CONTENTS=$(<$NEXT_SCRIPT)


    if [[ $NEXT_CONTENTS != $EMPTY  ]]
    then
        # Send notification
        TITLE="Starting next job 🚀"
        MESSAGE="Starting queued jobs on $HOSTNAME. No more jobs are queued ."
        curl -G "https://api2.pushdeer.com/message/push" \
         --data-urlencode "pushkey=$PUSHDEER_KEY" \
         --data-urlencode "text=$TITLE" \
         --data-urlencode "desp=$MESSAGE" \
         --data-urlencode "type=markdown"
        echo -e "\n"
        #curl 'https://api.pushover.net/1/messages.json' -X POST -d "token=$NOTIFICATION_TOKEN&user=$NOTIFICATION_USER&message=$MESSAGE&title=$TITLE"
    fi

done


# Send notification
TITLE="Job's done 👹"
MESSAGE="All jobs finished on host $HOSTNAME"
echo "https://api2.pushdeer.com/message/push?pushkey=$PUSHDEER_KEY&text=$TITLE&desp=$MESSAGE&type=markdown"
curl -G "https://api2.pushdeer.com/message/push" \
     --data-urlencode "pushkey=$PUSHDEER_KEY" \
     --data-urlencode "text=$TITLE" \
     --data-urlencode "desp=$MESSAGE" \
     --data-urlencode "type=markdown"
echo -e "\n"
#curl 'https://api.pushover.net/1/messages.json' -X POST -d "token=$NOTIFICATION_TOKEN&user=$NOTIFICATION_USER&message=$MESSAGE&title=$TITLE"
