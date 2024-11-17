const YT_ID_REGEX = /(?:youtu\.be\/|youtube\.com(?:\/embed\/|\/v\/|\/watch\?v=|\/user\/\S+|\/ytscreeningroom\?v=|\/sandalsResorts#\w\/\w\/.*\/))([^\/&]{10,12})/;

const preview_mode = typeof eel === 'undefined'

const player = new Plyr('#player');

player.on('ready', _ => {
    console.log("Player ready");
    console.log(player);
});

player.on('play', event => {
    
});

player.on('enterfullscreen', event => {
    let plyrVideoWrapper = document.getElementsByClassName('plyr__video-wrapper')
    console.log(plyrVideoWrapper)
    if (plyrVideoWrapper) {
        let myPluginCollection = document.getElementsByClassName('fullscreen-visible')
        if (myPluginCollection) {
            plyrVideoWrapper[0].appendChild(myPluginCollection[0])
        }
    }
});


const extractYtId = (url) => {
    const res = url.match(YT_ID_REGEX);
    if (res == null || res?.length < 2) {
        return null;
    }
    return res[1];
}

const get_data = async () => {
    const data = await eel.get_data()();
    console.log(data);
    document.querySelector("#test").innerHTML = data;
}

const handleUrlChange = (url) => {
    const player_wrapper = document.querySelector("#player-wrapper");
    const yt_id = extractYtId(url);
    console.log(yt_id);

    if (yt_id == null) {
        console.log("Invalid URL");
        return;
    }
  
    player.source = {
        type: 'video',
        sources: [
            {
                src: yt_id,
                provider: 'youtube',
            },
        ],
    };

    player_wrapper.classList.remove("hidden");

}

document.querySelector(".url-input").addEventListener("keyup", (event) => handleUrlChange(event.target.value));

document.querySelector("#toggle").addEventListener('change', (event) => {
    const label = document.querySelector(".switch-label");
    const boredomElements = document.querySelectorAll('[data-mode="boredom"]');
    const attentionElements = document.querySelectorAll('[data-mode="attention"]');

    if (event.target.checked) {
        label.innerHTML = "Attention Mode";

        boredomElements.forEach((el) => {
            console.log(el);
            el.classList.add('hidden');
        });
        attentionElements.forEach((el) => {
            el.classList.remove('hidden');
        });
    } else {
        label.innerHTML = "Boredom Mode";
        
        boredomElements.forEach((el) => {
            el.classList.remove('hidden');
        });
        attentionElements.forEach((el) => {
            el.classList.add('hidden');
        });
       
    }
});

document.querySelector('.fullscreen-visible[data-mode="attention"]').addEventListener('click', (event) => {
    player.currentTime -= 5;
    const popup = document.querySelector('.fullscreen-visible[data-mode="attention"]');
    popup.classList.add('transparent');
});

document.querySelector('.fullscreen-visible[data-mode="boredom"]').addEventListener('click', (event) => {
    player.speed = 2;
    const popup = document.querySelector('.fullscreen-visible[data-mode="boredom"]');
    popup.classList.add('transparent');
});

const start_prediction = () => {
    setInterval(async () => {
        const prediction = await eel.get_prediction()();
        console.log(prediction);

        if (prediction == "nudny") {
            if (player.speed == 1) {
                document.querySelector('.fullscreen-visible[data-mode="boredom"]').classList.remove('transparent');
            }
            document.querySelector('.fullscreen-visible[data-mode="attention"]').classList.remove('transparent');
        } else if (prediction == "ciekawy") {
            document.querySelector('.fullscreen-visible[data-mode="boredom"]').classList.add('transparent');
            document.querySelector('.fullscreen-visible[data-mode="attention"]').classList.add('transparent');
            player.speed = 1;
        }
    }, 1000);
}

const show_device_info = async () => {
    const device_info = document.querySelector("#device-info")
    
    const add_device = (device_name) => {
        const newDiv = document.createElement("div");
        newDiv.addEventListener("click", () => {
            console.log(device_name);
            if(!preview_mode) {
                eel.choose_device(device_name);
                start_prediction();
            }
            document.querySelector('.content[data-device="false"]').classList.add("display_none");
            document.querySelector('.content[data-device="true"]').classList.remove("display_none");
        });
        newDiv.innerHTML = device_name;
        device_info.appendChild(newDiv);
    }

    let devices;
    if (!preview_mode) {
        devices = await eel.get_devices()();
    } else {
        devices = await new Promise((resolve, reject) => {
            setTimeout(() => {
                resolve([
                    'Device 1',
                    'Device 2',
                    'Device 3',
                    'Device 4'
                ]);
            }, 2000);
        });
    }
    document.querySelector('.spinner-wrapper').classList.add('display_none');
    console.log(devices);
    devices.forEach(device => {
        add_device(device);
    });
}

show_device_info();


console.log("Main.js loaded");

