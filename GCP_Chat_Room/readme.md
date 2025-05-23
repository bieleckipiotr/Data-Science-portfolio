# ☁️ Cloud Messenger App with Secure Chatrooms

This project is a cloud-native **messenger application** built entirely on **Google Cloud Platform (GCP)** using **Terraform** for infrastructure automation. The app allows users to create and join secure, private chatrooms where only chat participants can view and send messages.

Designed with scalability, access control, and modularity in mind, this project showcases a modern, serverless messaging architecture.

---

## 🚀 Features

- **Secure Chatrooms**: Messages are scoped to chatroom members only.
- **Real-time Communication**: Built on **Firebase Realtime Database** and **Firestore** for fast, event-driven messaging.
- **Strict Role-Based Access Control**: Security rules restrict all read/write access to authorized users only.
- **Infrastructure as Code**: All GCP resources are provisioned and managed using **Terraform**.
- **Frontend Logic**: Built using **Svelte**, enabling fast, reactive UI and efficient API interaction.

---

## 🧱 Infrastructure & GCP Services Used

The app provisions and uses the following GCP services via [Terraform](https://www.terraform.io/):

- **Cloud Resource Manager API**
- **Service Usage API**
- **Firebase Management API**
- **Firebase Realtime Database API**
- **Firebase Storage & Cloud Storage APIs**:
  - `firebaserules.googleapis.com`
  - `firebasestorage.googleapis.com`
  - `storage.googleapis.com`
- **Firestore APIs**:
  - `firestore.googleapis.com`
  - `firebaserules.googleapis.com`
- **Firebase Web App Resource**
- **Firestore Database + Indexes**
- **Firebase Security Rules for Firestore**

---

## 🔐 Firestore Security Rules

The following rules enforce data access control:

| Collection/Subcollection     | Permission Rules                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------|
| `chatsMessages`             | Only authenticated users who are **chatroom members** can read or create messages |
| `userEmailToUid`            | Authenticated users can **read and create their own records**, no updates/deletes |
| `chatRooms`                 | Authenticated users can **read and create chatrooms**; only members can update     |
| `captions`, `memberNames`   | Access is limited to chat members, with appropriate scoped permissions            |

---

## 🛠️ Tech Stack

- **Terraform** – Infrastructure automation
- **Google Cloud Platform** – Backend services
- **Firebase (Realtime DB & Firestore)** – Real-time messaging and user data storage
- **Cloud Storage** – File storage if needed
- **Svelte** – Frontend/backend logic